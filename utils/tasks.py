import os
import pickle
from collections import Counter
from tqdm import tqdm
from typing import Dict, Callable, List, Optional
from pyhealth.datasets import BaseEHRDataset, SampleBaseDataset
from pyhealth.data import Patient, Visit
from pyhealth.datasets.utils import list_nested_levels, flatten_list
from pyhealth.tasks.length_of_stay_prediction import categorize_los


class SampleEHRDataset(SampleBaseDataset):
    """Sample EHR dataset class.

    This class inherits from `SampleBaseDataset` and is specifically designed
        for EHR datasets.

    Args:
        samples: a list of samples, each sample is a dict with
            patient_id, visit_id, and other task-specific attributes as key.
        dataset_name: the name of the dataset. Default is None.
        task_name: the name of the task. Default is None.

    Currently, the following types of attributes are supported:
        - a single value. Type: int/float/str. Dim: 0.
        - a single vector. Type: int/float. Dim: 1.
        - a list of codes. Type: str. Dim: 2.
        - a list of vectors. Type: int/float. Dim: 2.
        - a list of list of codes. Type: str. Dim: 3.
        - a list of list of vectors. Type: int/float. Dim: 3.

    Attributes:
        input_info: Dict, a dict whose keys are the same as the keys in the
            samples, and values are the corresponding input information:
            - "type": the element type of each key attribute, one of float, int, str.
            - "dim": the list dimension of each key attribute, one of 0, 1, 2, 3.
            - "len": the length of the vector, only valid for vector-based attributes.
        patient_to_index: Dict[str, List[int]], a dict mapping patient_id to
            a list of sample indices.
        visit_to_index: Dict[str, List[int]], a dict mapping visit_id to a list
            of sample indices.

    Examples:
        >>> from pyhealth.datasets import SampleEHRDataset
        >>> samples = [
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-0",
        ...             "single_vector": [1, 2, 3],
        ...             "list_codes": ["505800458", "50580045810", "50580045811"],  # NDC
        ...             "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
        ...             "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],  # ATC-4
        ...             "list_list_vectors": [
        ...                 [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
        ...                 [[7.7, 8.5, 9.4]],
        ...             ],
        ...             "label": 1,
        ...         },
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-1",
        ...             "single_vector": [1, 5, 8],
        ...             "list_codes": [
        ...                 "55154191800",
        ...                 "551541928",
        ...                 "55154192800",
        ...                 "705182798",
        ...                 "70518279800",
        ...             ],
        ...             "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7]],
        ...             "list_list_codes": [["A04A", "B035", "C129"], ["A07B", "A07C"]],
        ...             "list_list_vectors": [
        ...                 [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6]],
        ...                 [[7.7, 8.4, 1.3]],
        ...             ],
        ...             "label": 0,
        ...         },
        ...     ]
        >>> dataset = SampleEHRDataset(samples=samples)
        >>> dataset.input_info
        {'patient_id': {'type': <class 'str'>, 'dim': 0}, 'visit_id': {'type': <class 'str'>, 'dim': 0}, 'single_vector': {'type': <class 'int'>, 'dim': 1, 'len': 3}, 'list_codes': {'type': <class 'str'>, 'dim': 2}, 'list_vectors': {'type': <class 'float'>, 'dim': 2, 'len': 3}, 'list_list_codes': {'type': <class 'str'>, 'dim': 3}, 'list_list_vectors': {'type': <class 'float'>, 'dim': 3, 'len': 3}, 'label': {'type': <class 'int'>, 'dim': 0}}
        >>> dataset.patient_to_index
        {'patient-0': [0, 1]}
        >>> dataset.visit_to_index
        {'visit-0': [0], 'visit-1': [1]}
    """

    def __init__(self, samples: List[Dict], code_vocs=None, dataset_name="", task_name="", ignored_keys=[]):
        super().__init__(samples, dataset_name, task_name)
        self.samples = samples
        self.code_vocs = code_vocs
        self.ignored_keys = ignored_keys
        self.input_info: Dict = self._validate()
        self.patient_to_index: Dict[str, List[int]] = self._index_patient()
        self.visit_to_index: Dict[str, List[int]] = self._index_visit()
        self.type_ = "ehr"

    def _validate(self) -> Dict:
        """Helper function which validates the samples.

        Will be called in `self.__init__()`.

        Returns:
            input_info: Dict, a dict whose keys are the same as the keys in the
                samples, and values are the corresponding input information:
                - "type": the element type of each key attribute, one of float,
                    int, str.
                - "dim": the list dimension of each key attribute, one of 0, 1, 2, 3.
                - "len": the length of the vector, only valid for vector-based
                    attributes.
        """
        """ 1. Check if all samples are of type dict. """
        assert all(
            [isinstance(s, dict) for s in self.samples],
        ), "Each sample should be a dict"
        keys = self.samples[0].keys()

        """ 2. Check if all samples have the same keys. """
        assert all(
            [set(s.keys()) == set(keys) for s in self.samples]
        ), "All samples should have the same keys"

        """ 3. Check if "patient_id" and "visit_id" are in the keys."""
        assert "patient_id" in keys, "patient_id should be in the keys"
        assert "visit_id" in keys, "visit_id should be in the keys"

        """
        4. For each key, check if it is either:
            - a single value
            - a single vector
            - a list of codes
            - a list of vectors
            - a list of list of codes
            - a list of list of vectors
        Note that a value is either float, int, or str; a vector is a list of float 
        or int; and a code is str.
        """
        # record input information for each key
        input_info = {}
        for key in keys:
            if key in self.ignored_keys:
                continue
            """
            4.1. Check nested list level: all samples should either all be
            - a single value (level=0)
            - a single vector (level=1)
            - a list of codes (level=1)
            - a list of vectors (level=2)
            - a list of list of codes (level=2)
            - a list of list of vectors (level=3)
            """
            levels = set([list_nested_levels(s[key]) for s in self.samples])
            assert (
                len(levels) == 1 and len(list(levels)[0]) == 1
            ), f"Key {key} has mixed nested list levels across samples"
            level = levels.pop()[0]
            assert level in [
                0,
                1,
                2,
                3,
            ], f"Key {key} has unsupported nested list level across samples"

            # flatten the list
            if level == 0:
                flattened_values = [s[key] for s in self.samples]
            elif level == 1:
                flattened_values = [i for s in self.samples for i in s[key]]
            elif level == 2:
                flattened_values = [j for s in self.samples for i in s[key] for j in i]
            else:
                flattened_values = [
                    k for s in self.samples for i in s[key] for j in i for k in j
                ]

            """
            4.2. Check type: the basic type of each element should be float, 
            int, or str.
            """
            types = set([type(v) for v in flattened_values])
            assert (
                types == set([str]) or len(types.difference(set([int, float]))) == 0
            ), f"Key {key} has mixed or unsupported types ({types}) across samples"
            type_ = types.pop()
            """
            4.3. Combined level and type check.
            """
            if level == 0:
                # a single value
                input_info[key] = {"type": type_, "dim": 0}
            elif level == 1:
                # a single vector or a list of codes
                if type_ in [float, int]:
                    # a single vector
                    lens = set([len(s[key]) for s in self.samples])
                    assert len(lens) == 1, f"Key {key} has vectors of different lengths"
                    input_info[key] = {"type": type_, "dim": 1, "len": lens.pop()}
                else:
                    # a list of codes
                    # note that dim is different from level here
                    input_info[key] = {"type": type_, "dim": 2}
            elif level == 2:
                # a list of vectors or a list of list of codes
                if type_ in [float, int]:
                    lens = set([len(i) for s in self.samples for i in s[key]])
                    assert len(lens) == 1, f"Key {key} has vectors of different lengths"
                    input_info[key] = {"type": type_, "dim": 2, "len": lens.pop()}
                else:
                    # a list of list of codes
                    # note that dim is different from level here
                    input_info[key] = {"type": type_, "dim": 3}
            else:
                # a list of list of vectors
                assert type_ in [
                    float,
                    int,
                ], f"Key {key} has unsupported type across samples"
                lens = set([len(j) for s in self.samples for i in s[key] for j in i])
                assert len(lens) == 1, f"Key {key} has vectors of different lengths"
                input_info[key] = {"type": type_, "dim": 3, "len": lens.pop()}

        return input_info

    def _index_patient(self) -> Dict[str, List[int]]:
        """Helper function which indexes the samples by patient_id.

        Will be called in `self.__init__()`.
        Returns:
            patient_to_index: Dict[str, int], a dict mapping patient_id to a list
                of sample indices.
        """
        patient_to_index = {}
        for idx, sample in enumerate(self.samples):
            patient_to_index.setdefault(sample["patient_id"], []).append(idx)
        return patient_to_index

    def _index_visit(self) -> Dict[str, List[int]]:
        """Helper function which indexes the samples by visit_id.

        Will be called in `self.__init__()`.

        Returns:
            visit_to_index: Dict[str, int], a dict mapping visit_id to a list
                of sample indices.
        """
        visit_to_index = {}
        for idx, sample in enumerate(self.samples):
            visit_to_index.setdefault(sample["visit_id"], []).append(idx)
        return visit_to_index

    @property
    def available_keys(self) -> List[str]:
        """Returns a list of available keys for the dataset.

        Returns:
            List of available keys.
        """
        keys = self.samples[0].keys()
        return list(keys)

    def get_distribution_tokens(self, key: str) -> Dict[str, int]:
        """Gets the distribution of tokens with a specific key in the samples.

        Args:
            key: the key of the tokens in the samples.

        Returns:
            distribution: a dict mapping token to count.
        """

        tokens = self.get_all_tokens(key, remove_duplicates=False, sort=False)
        counter = Counter(tokens)
        return counter

    def stat(self) -> str:
        """Returns some statistics of the task-specific dataset."""
        lines = list()
        lines.append(f"Statistics of sample dataset:")
        lines.append(f"\t- Dataset: {self.dataset_name}")
        lines.append(f"\t- Task: {self.task_name}")
        lines.append(f"\t- Number of samples: {len(self)}")
        num_patients = len(set([sample["patient_id"] for sample in self.samples]))
        lines.append(f"\t- Number of patients: {num_patients}")
        num_visits = len(set([sample["visit_id"] for sample in self.samples]))
        lines.append(f"\t- Number of visits: {num_visits}")
        lines.append(
            f"\t- Number of visits per patient: {len(self) / num_patients:.4f}"
        )
        for key in self.samples[0]:
            if key in ["patient_id", "visit_id"]:
                continue
            if key in self.ignored_keys:
                continue
            input_type = self.input_info[key]["type"]
            input_dim = self.input_info[key]["dim"]

            if input_dim <= 1:
                # a single value or vector
                num_events = [1 for sample in self.samples]
            elif input_dim == 2:
                # a list
                num_events = [len(sample[key]) for sample in self.samples]
            elif input_dim == 3:
                # a list of list
                num_events = [len(flatten_list(sample[key])) for sample in self.samples]
            else:
                raise NotImplementedError
            lines.append(f"\t- {key}:")
            lines.append(
                f"\t\t- Number of {key} per sample: "
                f"{sum(num_events) / len(num_events):.4f}"
            )
            if input_type == str or input_dim == 0:
                pass
                # single value or code-based
                # lines.append(
                #     f"\t\t- Number of unique {key}: {len(self.get_all_tokens(key))}"
                # )
                # distribution = self.get_distribution_tokens(key)
                # top10 = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[
                #     :10
                # ]
                # lines.append(f"\t\t- Distribution of {key} (Top-10): {top10}")
            else:
                # vector-based
                vector = self.samples[0][key]
                lines.append(f"\t\t- Length of {key}: {self.input_info[key]['len']}")
        print("\n".join(lines))
        return "\n".join(lines)


def set_task(dataset: BaseEHRDataset, task_fn: Callable, dataset_name, task_name: Optional[str] = None, ignored_keys = [], ids=None) -> SampleEHRDataset:
    if task_name is None:
        task_name = task_fn.__name__
    if os.path.exists(f"export/{dataset_name}/{task_name}.pkl"):
        sample_dataset = pickle.load(
            open(f"export/{dataset_name}/{task_name}.pkl", "rb")
        )
        return sample_dataset
    if "mimiciv" in dataset_name and ids is None:
        ids = set(pickle.load(open('resources/mimiciv_visitid.pkl', 'rb')))
    samples = []
    for patient_id, patient in tqdm(
        dataset.patients.items(), desc=f"Generating samples for {task_name}"
    ):
        if "mimiciv" in dataset_name:
            samples.extend(task_fn(patient, ids=ids))
        else:
            samples.extend(task_fn(patient, ids=ids))

    sample_dataset = SampleEHRDataset(
        samples=samples,
        dataset_name=dataset.dataset_name,
        task_name=task_name,
        ignored_keys=ignored_keys
    )
    with open(f"export/{dataset_name}/{task_name}.pkl", "wb") as f:
        pickle.dump(sample_dataset, f)
    return sample_dataset


def mortality_prediction_mimic3_fn(patient: Patient, ids=None):
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # exclude: visits without condition, procedure, and drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        data = {
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": [conditions],
            "procedures": [procedures],
            "drugs": [drugs],
            "label": mortality_label,
        }
        samples.append(data)
    return samples


def readmission_prediction_mimic3_fn(patient: Patient, time_window=30, ids=None):
    samples = []
    
    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.discharge_time).days
        readmission_label = 1 if time_diff < time_window else 0

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        data = {
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": [conditions],
            "procedures": [procedures],
            "drugs": [drugs],
            "label": readmission_label,
        }
        samples.append(data)
    return samples


def length_of_stay_prediction_mimic3_fn(patient: Patient, ids=None):
    samples = []

    # for visit in patient:
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)
        data = {
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": [conditions],
            "procedures": [procedures],
            "drugs": [drugs],
            "label": los_category,
        }
        samples.append(data)
    return samples


def mortality_prediction_mimic4_fn(patient: Patient, ids=None):
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        if ids is not None and visit.visit_id not in ids:
            continue
        next_visit: Visit = patient[i + 1]

        if visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, and drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        data = {
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": [conditions],
            "procedures": [procedures],
            "drugs": [drugs],
            "label": mortality_label,
        }
        samples.append(data)
    return samples


def readmission_prediction_mimic4_fn(patient: Patient, time_window=30, ids=None):
    samples = []
    
    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]
        if ids is not None and visit.visit_id not in ids:
            continue
        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.discharge_time).days
        readmission_label = 1 if time_diff < time_window else 0

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        data = {
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": [conditions],
            "procedures": [procedures],
            "drugs": [drugs],
            "label": readmission_label,
        }
        samples.append(data)
    return samples


def length_of_stay_prediction_mimic4_fn(patient: Patient, ids=None):
    samples = []
    # for visit in patient:
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        if ids is not None and visit.visit_id not in ids:
            continue
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)
        data = {
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": [conditions],
            "procedures": [procedures],
            "drugs": [drugs],
            "label": los_category,
        }
        samples.append(data)
    return samples


def read_text(dataset, text_path=None, text_num=0, perplexities=None, text_embedding=None, return_text=False):
    for i in range(len(dataset)):
        visit_id = dataset[i]["visit_id"]
        text = []
        perplexity = []
        embedding = []
        if text_num == 0:
            if return_text:
                with open(os.path.join(text_path, f'{visit_id}.txt'), 'r') as f:
                    text.append(''.join(f.readlines()))
            perplexity = None
            embedding = text_embedding[visit_id] if text_embedding is not None else None
        else:
            for j in range(text_num):
                if return_text:
                    with open(os.path.join(text_path, f'{visit_id}-{j}.txt'), 'r') as f:
                        text.append(''.join(f.readlines()))
                perplexity.append(perplexities[visit_id][j])
                if text_embedding is not None:
                    embedding.append(text_embedding[visit_id][j])
                else:
                    embedding = None
        if return_text:
            dataset[i]["text"] = text
        if perplexity is not None:
            dataset[i]["perplexity"] = perplexity
        if embedding is not None:
            dataset[i]["embedding"] = embedding
    return dataset


def get_name(visit: Visit, table, key="name"):
    events = visit.get_event_list(table)
    return [event.attr_dict[table][key] for event in events]


def get_drug_name(visit: Visit, table="PRESCRIPTIONS"):
    events = visit.get_event_list(table)
    max_day = (visit.discharge_time - visit.encounter_time).days
    days = [int((event.timestamp - visit.encounter_time).days) if event.timestamp is not None else None for event in events]
    for i in range(len(days)):
        if days[i] is None:
            # assert i > 0
            days[i] = days[i - 1]
        elif days[i] < 0:
            days[i] = 0
        elif days[i] > max_day:
            days[i] = max_day
            
    # return days, [f"{event.attr_dict['dose_val'] if 'dose_val' in event.attr_dict else ''}{event.attr_dict['dose_unit'] if 'dose_unit' in event.attr_dict else ''} {event.attr_dict['drug']}" for event in events]
    return days, [event.attr_dict['drug'] for event in events]


def calculate_age(current_date, birth_date):
    age = current_date.year - birth_date.year - ((current_date.month, current_date.day) < (birth_date.month, birth_date.day))
    return age


def mimic3_fn_for_generation(patient: Patient, time_window=30, ids=None):
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        if ids is not None and visit.visit_id not in ids:
            continue
        next_visit: Visit = patient[i + 1]
        time_diff = (next_visit.encounter_time - visit.discharge_time).days
        readmission_label = 1 if time_diff < time_window else 0
        mortality_label = 0 if next_visit.discharge_status not in [0, 1] else int(next_visit.discharge_status)
        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        
        age = calculate_age(visit.encounter_time, patient.birth_datetime)
        condition_names = get_name(visit, "DIAGNOSES_ICD")
        procedure_names = get_name(visit, "PROCEDURES_ICD")
        drug_days, drug_names = get_drug_name(visit)
        drug_names = list(set(drug_names))
        drug_names1 = []
        for drug in drug_names:
            if "NS" not in drug and "D5W" not in drug:
                drug_names1.append(drug)
        drug_names = drug_names1
        # exclude: visits without condition, procedure, and drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "age": age,
                "gender": patient.gender,
                "diagnosis": visit.attr_dict['diagnosis'],
                "condition_names": [condition_names],
                "procedure_names": [procedure_names],
                "drug_days": [drug_days],
                "drug_names": [drug_names],
                "readmission_label": readmission_label,
                "mortality_label": mortality_label,
                "los_days": los_days
            }
        )
    # no cohort selection
    return samples


def mimic4_fn_for_generation(patient: Patient, time_window=30, ids=None):
    samples = []
    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        if ids is not None and visit.visit_id not in ids:
            continue
        next_visit: Visit = patient[i + 1]
        time_diff = (next_visit.encounter_time - visit.discharge_time).days
        readmission_label = 1 if time_diff < time_window else 0
        mortality_label = 0 if next_visit.discharge_status not in [0, 1] else int(next_visit.discharge_status)
        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        conditions = [event.code + "ICD9" if event.attr_dict["diagnoses_icd"]["icd_version"] == 9 else event.code + "ICD10" for event in visit.get_event_list("diagnoses_icd")]
        conditions = list(dict.fromkeys(conditions))
        procedures = [event.code + "ICD9" if event.attr_dict["procedures_icd"]["icd_version"] == 9 else event.code + "ICD10" for event in visit.get_event_list("procedures_icd")]
        procedures = list(dict.fromkeys(procedures))
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, and drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        if not 30 <= len(conditions) + len(procedures) + len(drugs) <= 70:
            continue
        
        age = calculate_age(visit.encounter_time, patient.birth_datetime)
        condition_names = get_name(visit, "diagnoses_icd", "long_title")
        procedure_names = get_name(visit, "procedures_icd", "long_title")
        drug_days, drug_names = get_drug_name(visit, "prescriptions")
        drug_names = list(set(drug_names))
        drug_names1 = []
        for drug in drug_names:
            if "NS" not in drug and "D5W" not in drug:
                drug_names1.append(drug)
        drug_names = drug_names1
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "age": age,
                "gender": patient.gender,
                "diagnosis": None,
                "condition_names": [condition_names],
                "procedure_names": [procedure_names],
                "drug_days": [drug_days],
                "drug_names": [drug_names],
                "readmission_label": readmission_label,
                "mortality_label": mortality_label,
                "los_days": los_days
            }
        )
    # no cohort selection
    return samples

