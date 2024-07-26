import os
from typing import Dict
from tqdm import tqdm
import pandas as pd

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from pyhealth.datasets.utils import strptime


class MIMIC3Dataset(MIMIC3Dataset):
    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PATIENTS and ADMISSIONS tables.

        Will be called in `self.parse_tables()`

        Docs:
            - PATIENTS: https://mimic.mit.edu/docs/iii/tables/patients/
            - ADMISSIONS: https://mimic.mit.edu/docs/iii/tables/admissions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id which is updated with the mimic-3 table result.

        Returns:
            The updated patients dict.
        """
        # read patients table
        patients_df = pd.read_csv(
            os.path.join(self.root, "PATIENTS.csv"),
            dtype={"SUBJECT_ID": str},
            nrows=1000 if self.dev else None,
        )
        # read admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "ADMISSIONS.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str},
        )
        # merge patient and admission tables
        df = pd.merge(patients_df, admissions_df, on="SUBJECT_ID", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["SUBJECT_ID", "ADMITTIME", "DISCHTIME"], ascending=True)
        # group by patient
        df_group = df.groupby("SUBJECT_ID")

        # parallel unit of basic information (per patient)
        def basic_unit(p_id, p_info):
            patient = Patient(
                patient_id=p_id,
                birth_datetime=strptime(p_info["DOB"].values[0]),
                death_datetime=strptime(p_info["DOD_HOSP"].values[0]),
                gender=p_info["GENDER"].values[0],
                ethnicity=p_info["ETHNICITY"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("HADM_ID", sort=False):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=strptime(v_info["ADMITTIME"].values[0]),
                    discharge_time=strptime(v_info["DISCHTIME"].values[0]),
                    discharge_status=v_info["HOSPITAL_EXPIRE_FLAG"].values[0],
                    insurance=v_info["INSURANCE"].values[0],
                    language=v_info["LANGUAGE"].values[0],
                    religion=v_info["RELIGION"].values[0],
                    marital_status=v_info["MARITAL_STATUS"].values[0],
                    ethnicity=v_info["ETHNICITY"].values[0],
                    diagnosis=v_info["DIAGNOSIS"].values[0]
                )
                # add visit
                patient.add_visit(visit)
            return patient

        # parallel apply
        df_group = df_group.parallel_apply(
            lambda x: basic_unit(x.SUBJECT_ID.unique()[0], x)
        )
        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients

    def parse_prescriptions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PRESCRIPTIONS table.

        Will be called in `self.parse_tables()`

        Docs:
            - PRESCRIPTIONS: https://mimic.mit.edu/docs/iii/tables/prescriptions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "PRESCRIPTIONS"
        self.code_vocs["drugs"] = "NDC"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            # dtype={"SUBJECT_ID": str, "HADM_ID": str, "NDC": str},
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "NDC": str, "DRUG": str, "DOSE_VAL_RX": str, "DOSE_UNIT_RX": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "NDC"])
        # sort by start date and end date
        df = df.sort_values(
            ["SUBJECT_ID", "HADM_ID", "STARTDATE", "ENDDATE"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit for prescription (per patient)
        def prescription_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for timestamp, code, drug, dose_val, dose_unit in zip(v_info["STARTDATE"], v_info["NDC"], v_info["DRUG"], v_info["DOSE_VAL_RX"], v_info["DOSE_UNIT_RX"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="NDC",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                        drug=drug
                    )
                    if isinstance(dose_val, str):
                        event.attr_dict['dose_val'] = dose_val
                    if isinstance(dose_unit, str):
                        event.attr_dict['dose_unit'] = dose_unit
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: prescription_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def load_name_mapping(self, name_dict):
        # Since code mapping may map one event into more than one events, it cannot be activated with name mapping
        assert (len(name_dict) == 0) ^ (len(self.code_mapping) == 0)
        self.name_mapping = {}
        for s_vocab, t_vocab in name_dict.items():
            df = pd.read_csv(
                f"resources/{t_vocab}.csv",
            )
            for index, row in df.iterrows():
                df.loc[index, 'code'] = row['code'].replace('.', '')
            df.set_index('code', inplace=True)
            dictionary = df.to_dict(orient='index')
            self.name_mapping[s_vocab] = dictionary
    
    def convert_name_in_patient_dict(self):
        for p_id, patient in tqdm(self.patients.items(), desc="Mapping names"):
            for visit in patient:
                for table in visit.available_tables:
                    for event in visit.get_event_list(table):
                        if event.table in self.name_mapping:
                            if event.code in self.name_mapping[event.table]:
                                event.attr_dict[table] = self.name_mapping[event.table][event.code]
                            elif event.table == "DIAGNOSES_ICD" and event.code == "71970":
                                event.attr_dict[table] = self.name_mapping[event.table]["7197"]
                            else:
                                print(event.table, event.code)
                                raise ValueError(f"Unknown code {event.code} in table {event.table}")


class MIMIC4Dataset(MIMIC4Dataset): 
    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper functions which parses patients and admissions tables.

        Will be called in `self.parse_tables()`

        Docs:
            - patients:https://mimic.mit.edu/docs/iv/modules/hosp/patients/
            - admissions: https://mimic.mit.edu/docs/iv/modules/hosp/admissions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        # read patients table
        patients_df = pd.read_csv(
            os.path.join(self.root, "patients.csv"),
            dtype={"subject_id": str},
            nrows=1000 if self.dev else None,
        )
        # read admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "admissions.csv"),
            dtype={"subject_id": str, "hadm_id": str},
        )
        # merge patients and admissions tables
        df = pd.merge(patients_df, admissions_df, on="subject_id", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["subject_id", "admittime", "dischtime"], ascending=True)
        # group by patient
        df_group = df.groupby("subject_id")

        # parallel unit of basic information (per patient)
        def basic_unit(p_id, p_info):
            # no exact birth datetime in MIMIC-IV
            # use anchor_year and anchor_age to approximate birth datetime
            anchor_year = int(p_info["anchor_year"].values[0])
            anchor_age = int(p_info["anchor_age"].values[0])
            birth_year = anchor_year - anchor_age
            patient = Patient(
                patient_id=p_id,
                # no exact month, day, and time, use Jan 1st, 00:00:00
                birth_datetime=strptime(str(birth_year)),
                # no exact time, use 00:00:00
                death_datetime=strptime(p_info["dod"].values[0]),
                gender=p_info["gender"].values[0],
                ethnicity=p_info["race"].values[0],
                anchor_year_group=p_info["anchor_year_group"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("hadm_id", sort=False):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=strptime(v_info["admittime"].values[0]),
                    discharge_time=strptime(v_info["dischtime"].values[0]),
                    discharge_status=v_info["hospital_expire_flag"].values[0],
                )
                # add visit
                patient.add_visit(visit)
            return patient

        # parallel apply
        df_group = df_group.parallel_apply(
            lambda x: basic_unit(x.subject_id.unique()[0], x)
        )
        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients
    
    def parse_prescriptions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PRESCRIPTIONS table.

        Will be called in `self.parse_tables()`

        Docs:
            - PRESCRIPTIONS: https://mimic.mit.edu/docs/iii/tables/prescriptions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "prescriptions"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"subject_id": str, "hadm_id": str, "ndc": str, "drug": str, "dose_val_rx": str, "dose_unit_rx": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "ndc"])
        # sort by start date and end date
        df = df.sort_values(
            ["subject_id", "hadm_id", "starttime", "stoptime"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit for prescription (per patient)
        def prescription_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for timestamp, code, drug, dose_val, dose_unit in zip(v_info["starttime"], v_info["ndc"], v_info["drug"], v_info["dose_val_rx"], v_info["dose_unit_rx"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="NDC",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                        drug=drug
                    )
                    if isinstance(dose_val, str):
                        event.attr_dict['dose_val'] = dose_val
                    if isinstance(dose_unit, str):
                        event.attr_dict['dose_unit'] = dose_unit
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: prescription_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def load_name_mapping(self, name_dict):
        # Since code mapping may map one event into more than one events, it cannot be activated with name mapping
        assert (len(name_dict) == 0) ^ (len(self.code_mapping) == 0)
        self.name_mapping = {}
        for s_vocab, t_vocab in name_dict.items():
            df = pd.read_csv(
                f"{self.root + t_vocab}.csv",
                # index_col="code"
            )
            for index, row in df.iterrows():
                df.loc[index, 'icd_code'] = row['icd_code'] + 'ICD' + str(row['icd_version'])
            df.set_index('icd_code', inplace=True)
            dictionary = df.to_dict(orient='index')
            self.name_mapping[s_vocab] = dictionary
    
    def convert_name_in_patient_dict(self):
        for p_id, patient in tqdm(self.patients.items(), desc="Mapping names"):
            for visit in patient:
                for table in visit.available_tables:
                    for event in visit.get_event_list(table):
                        if event.table in self.name_mapping:
                            if event.code + 'ICD9' if '9' in event.vocabulary else event.code + 'ICD10' in self.name_mapping[event.table]:
                                event.attr_dict[table] = self.name_mapping[event.table][event.code + 'ICD9' if '9' in event.vocabulary else event.code + 'ICD10']
                            else:
                                # print(event.table, event.code + 'ICD9' if '9' in event.vocabulary else event.code + 'ICD10')
                                raise ValueError(f"Unknown code {event.code} in table {event.table}")
