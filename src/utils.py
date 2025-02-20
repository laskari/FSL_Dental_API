import os

DateKeys = ['DEN_DateAppliancePlaced', 'DEN_DateOfAccident', 'DEN_DatePriorPlacement', 'DEN_DOS','DEN_FacProvSignDate', 'DEN_PatDOB', 'DEN_PatSignDate', 'DEN_PriInsDOB', 'DEN_PriSignDate', 'DEN_SecInsDOB']
PhoneNumberKeys = ['DEN_BillProvPhone', 'DEN_FacProvPhone']
NPIKeys = ['DEN_BillProvNPI', 'DEN_FacProvNPI']
PostCode = ['DEN_BillProvPostCode', 'DEN_FacProvPostCode', 'DEN_PatPostCode',  'DEN_PayerPostCode', 'DEN_PriInsPostCode', 'DEN_SecInsPostCode']


def process_confidence_scores(text, confidence_scores, valid_lengths):
    # Ensure confidence_scores is a list
    if isinstance(confidence_scores, float):
        confidence_scores = [confidence_scores]  # Convert to a list if it's a single float
   
    for idx, each_t in enumerate(text):
        if each_t['value'] != '[BLANK]':
            if len(each_t['value']) not in valid_lengths:
                # If the length is not valid, halve the confidence score
                # print(f"Invalid value: {each_t['value']} (Length: {len(each_t['value'])})")
                confidence_scores[idx] = confidence_scores[idx] / 2  # Halve the confidence score for invalid length
            else:
                # print(f"Valid value: {each_t['value']} (Length: {len(each_t['value'])})")
                pass
    return confidence_scores
 
 
def process_data(data):
    for each_key in data['result'].keys():
        text = data['result'][each_key]['text']
        confidence_scores = data['result'][each_key]['confidence_score']
       
        if each_key in DateKeys:
            valid_lengths = [6, 8]  # Only 6 and 8 are valid lengths
        elif each_key in PhoneNumberKeys or each_key in NPIKeys:
            valid_lengths = [10]
        elif each_key in PostCode:
            valid_lengths = [5, 9]
        else:
            continue  # Skip keys that don't match the known categories
       
        # Process confidence scores for the given key
        confidence_scores = process_confidence_scores(text, confidence_scores, valid_lengths)
       
        # Update the confidence_scores in the data structure
        data['result'][each_key]['confidence_score'] = confidence_scores
   
    return data  # Return the modified data
