import os
ROOT = os.getcwd()
VERSION = "ada_v2.1"
# ROOT = "/Data/FSL_codebase/FSL_Dental_API"
LOG_DIR = "logs"
LOG_FILE = "ada_logs.log"
artifact_path = 'artifacts'
LOGFILE_DIR = os.path.join(ROOT, LOG_DIR, LOG_FILE)

CATEGORY_MAPPING_PATH = os.path.join(ROOT, artifact_path, 'notes.json')

model_path = os.path.join(ROOT, artifact_path, 'ada__88.pth')

ADA_FORM_KEY_MAPPING = os.path.join(ROOT, artifact_path,"FSL_Forms_Keys.xlsx")

BBOX_DONUT_Mapping_Dict = {
    "1_Type_of_Transaction":["DEN_TransTypeStmtOfServ","DEN_TransTypeReqforPreAuth","DEN_TransTypeEPSDT",\
                            "ClaimLevel", "DEN_DentalDrop", "DEN_DentalForm", "DEN_DiagDesc", "DEN_FedTaxClass", "DEN_AMIField"],
    "2_Pre_Number": "DEN_PreAuthNum",
    "3_Company_address":["DEN_PayerAddr1","DEN_PayerCity","DEN_PayerPostCodeExt", "DEN_PayerOrgName",\
                         "DEN_PayerFName","DEN_PayerLName", "DEN_PayerPhoneNumber", "DEN_PayerState", "DEN_PayerPostCode"],
    "4_Other_Coverage":["DEN_IsThereSecInsN","DEN_IsThereSecInsY", "DEN_IsThereSecIns"],
    "5_Name": "DEN_SecInsFullName",
    "6_DOB":"DEN_SecInsDOB",
    "7_Gender":["DEN_SecInsSexF","DEN_SecInsSexM"],
    "8_SSN": "DEN_SecInsIDNumber",
    "9_Plan_Number":"DEN_SecInsPolGrpNumber",
    "10_Relationship":["DEN_SecRelShipDep","DEN_SecRelShipOther", "DEN_SecRelShipSelf", "DEN_SecRelShipSpouse"],
    "11_Company_or_plan": ["DEN_SecInsOrgName","DEN_SecInsAddr1","DEN_SecInsCity", "DEN_SecPostCodeExt", "DEN_SecPhoneNumber", \
                           "DEN_SecInsState", "DEN_SecInsPostCode"],
    "12_Policyholder_details":["DEN_PriInsAddr1", "DEN_PriInsCity","DEN_PriInsFullName", "DEN_PriInsState", \
                               "DEN_PriInsPostCode"],
    "13_DOB":"DEN_PriInsDOB",
    "14_Gender":["DEN_PriInsSexF","DEN_PriInsSexM"],
    "15_SSN":["DEN_PriInsIDNumber", "DEN_SSNField"],
    "16_Plan_Number":"DEN_PriInsPolGrpNumber",
    "17_Employer_Name":["DEN_PriInsPlanName","DEN_PriInsEmpName"],
    "18_Relationship":["DEN_PatRelShipDep","DEN_PatRelShipOther","DEN_PatRelShipSelf","DEN_PatRelShipSpouse"],
    "19_Use":["DEN_PatStudentStatusCodeFT", "DEN_PatStudentStatusCodePT"],
    "20_Name":["DEN_PatAddr1","DEN_PatCity","DEN_PatFullName","DEN_PatState", "DEN_PatPostCode"],
    "21_DOB":"DEN_PatDOB",
    "22_Gender":["DEN_PatSexF","DEN_PatSexM"],
    "23_Patient_ID":"DEN_PatAcctNo",
    "24_31_Table":["DEN_DOS","DEN_CavityArea","DEN_ToothSystem","DEN_ToothId","DEN_ToothSurface","DEN_ProcCode",\
                   "DEN_Units", "DEN_ProcDesc", "DEN_LineCharges", "DEN_DiagPtr", "SY_RowType"],
    "31_A_Other_Fee":"DEN_OtherCharges",
    "32_Total_Fee":"DEN_TotalCharges",
    "33_Missing_Teeth":"DEN_MissingTooth",
    "34_A_Diag_Codes":["21_ICDInd", "DEN_DiagCode"],
    "34_Code_list_Qualifier": "DEN_DiagCodeQual",
    "35_Remarks":["DEN_Remarks", "DEN_CHCCNum", "DEN_CHCNum", "DEN_DXCNum", "DEN_NEANum"],
    "36_Signature":["DEN_PatSignonFile", "DEN_PatSignDate"],
    "37_Signature":["DEN_PriSignonFile", "DEN_PriSignDate"],
    "38_Place_of_treatment":["DEN_PlaceOfTreatment", "DEN_FacProvHospital",\
                                "DEN_FacProvOffice", "DEN_FacProvOther","DEN_FacProvTreatmentECF"],
    "39_Enclosures":["DEN_Encloseure",  "DEN_NoOfDraftImages", "DEN_NoOfModels", "DEN_NoOfRadiographs",],
    "40_Orthodontics":["DEN_OrthodonticsN","DEN_OrthodonticsY"],
    "41_Date":"DEN_DateAppliancePlaced",
    "42_Months_remaining":"DEN_MnthsOfTrmntRemain",
    "43_Prosthesis":["DEN_RplProsthesisN","DEN_RplProsthesisY"],
    "44_Date":"DEN_DatePriorPlacement",
    "45_Treatment_resulting_form":["DEN_PatConditionAuto", "DEN_PatConditionEmp", "DEN_PatConditionOther"],
    "46_Date_of_accident":"DEN_DateOfAccident",
    "47_Auto_Accident_state":"DEN_AutoAccidentState" ,
    "48_Dentist_Address": ["DEN_BillProvAddr1", "DEN_BillProvCity", "DEN_BillProvFullName", "DEN_BillProvOrgName",\
                           "DEN_BillProvPrefix", "DEN_BillProvState", "DEN_BillProvSuffix", "DEN_BillProvPostCode", "DEN_BillOrgChk"],
    "49_NPI":"DEN_BillProvNPI",
    "50_Licence_Number": "DEN_BillProvLicenseId",
    "51_SSN_TIN": "DEN_BillProvFedIdCode",
    "52_A_Addl_Provider_id": ["DEN_FacProvOtherIdFull", "DEN_BillProvOtherId"],
    "52_Phone_Number":"DEN_BillProvPhone",
    "53_Signature": ["DEN_FacProvOrgName", "DEN_FacProvSignOnFile", "DEN_FacProvSignDate", "DEN_FacProvFullName"],
    "54_NPI": ["DEN_FacProvNPI", "DEN_BillFacNPIChk"],
    "55_Licence_Number": ["DEN_FacProvLicenseId", "DEN_BillFacLicenseChk"],
    "56_Address":["DEN_FacProvAddr1","DEN_FacProvCity", "DEN_FacProvState", "DEN_FacProvPostCode",\
                     "DEN_FacProvSpecialtyCode", "DEN_BillFacAddrChk", "DEN_BillFacSameChk"],
    "57_Phone_Number":"DEN_FacProvPhone",
    "58_Addl_provider_id":"DEN_FacProvOtherId"
}
