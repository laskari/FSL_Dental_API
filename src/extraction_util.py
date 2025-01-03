import json
import torch
import io
import torchvision
import pandas as pd
from torchvision.io import read_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from PIL import Image
from torchvision import transforms
import pandas as pd
from transformers import AutoProcessor, VisionEncoderDecoderModel
import requests
import json
from PIL import Image
import torch
import argparse
import os
import warnings
from tqdm import tqdm
from config import *
from src.logger import log_message


warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#key_mapping = pd.read_excel(ADA_FORM_KEY_MAPPING)
#mapping_dict = key_mapping.set_index('Key_Name').to_dict()['Modified_key']
#reverse_mapping_dict = {v: k for k, v in mapping_dict.items()}


class DentalRoiPredictor:
    def __init__(self, model_path, category_mapping_path=CATEGORY_MAPPING_PATH):
        self.category_mapping = self._load_category_mapping(category_mapping_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()

    def _load_model(self, model_path):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = len(self.category_mapping) + 1
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def _load_category_mapping(self, category_mapping_path):
        with open(category_mapping_path) as f:
            return {c['id'] + 1: c['name'] for c in json.load(f)['categories']}

    def _get_transforms(self):
        return T.Compose([T.ToDtype(torch.float, scale=True), T.ToPureTensor()])

    def _apply_nms(self, orig_prediction, iou_thresh=0.3):
        keep = torchvision.ops.nms(
            orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
        final_prediction = orig_prediction
        final_prediction['boxes'] = final_prediction['boxes'][keep]
        final_prediction['scores'] = final_prediction['scores'][keep]
        final_prediction['labels'] = final_prediction['labels'][keep]
        return final_prediction

    def _postprocessing_annotation(self, df):
        x0_missing_teeth = df.loc[df['class_name'] == '33_Missing_Teeth', 'x0'].mean()
        x1_other_fee = df.loc[df['class_name'] == '31_A_Other_Fee', 'x1'].mean()
        df.loc[df['class_name'] == '24_31_Table', 'x0'] = x0_missing_teeth
        df.loc[df['class_name'] == '24_31_Table', 'x1'] = x1_other_fee
        df.loc[df['class_name'] == '35_Remarks', 'x0'] = x0_missing_teeth
        df.loc[df['class_name'] == '35_Remarks', 'x1'] = x1_other_fee
        return df

    def predict_image(self, image):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval().to(device)
        # pil_image = Image.open(image_path)
        # to_tensor = transforms.ToTensor()
        # image = to_tensor(pil_image)
        image_tensor = self.transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = self.model(image_tensor)
        return predictions

    def predict_and_get_dataframe(self, image_path, image,  iou_thresh=0.5):
        predictions = self.predict_image(image)
        pred = predictions[0]
        pred_nms = self._apply_nms(pred, iou_thresh=iou_thresh)

        pred_dict = {
            'boxes': pred_nms['boxes'].cpu().numpy(),
            'labels': pred_nms['labels'].cpu().numpy(),
            'scores': pred_nms['scores'].cpu().numpy()
        }

        boxes_flat = pred_dict['boxes'].reshape(-1, 4)
        labels_flat = pred_dict['labels'].reshape(-1)
        scores_flat = pred_dict['scores'].reshape(-1)

        class_names = [self.category_mapping[label_id] for label_id in labels_flat]
        num_predictions = len(boxes_flat)
        # file_name = [image_path] * num_predictions
        file_name = [image_path.split(".")[0]] * num_predictions


        infer_df = pd.DataFrame({
            'file_name': file_name,
            'x0': boxes_flat[:, 0],
            'y0': boxes_flat[:, 1],
            'x1': boxes_flat[:, 2],
            'y1': boxes_flat[:, 3],
            'label': labels_flat,
            'class_name': class_names,
            'score': scores_flat
        })

        post_processed_df = self._postprocessing_annotation(infer_df)
        return post_processed_df

# Load the RPI model
frcnn_predictor = DentalRoiPredictor(MODEL_PATH)


def roi_model_inference(image_path, image):
    result_df = frcnn_predictor.predict_and_get_dataframe(image_path, image)
    max_score_indices = result_df.groupby('class_name')['score'].idxmax()
    result_df = result_df.loc[max_score_indices]
    return result_df

def run_prediction_donut(image, model, processor):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=2,
        epsilon_cutoff=6e-4,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        output_scores=True,
        return_dict_in_generate=True,
    )
    scores = outputs.scores 
    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = prediction.replace("<one>", "1")
    prediction = processor.token2json(prediction)
    return prediction, outputs, scores

def split_and_expand(row):
    try:
        if row['Key'] == "33_Missing_Teeth":
            keys = [row['Key']]
            values = row['Value'].split(';')[0]
        else:
            keys = [row['Key']] * len(row['Value'].split(';'))
            values = row['Value'].split(';')
        return pd.DataFrame({'Key': keys, 'Value': values})
    except Exception as e:
        log_message(f"Error while splitting {e}", level="ERROR")
        raise e

def load_model(device):
    try:
        processor = AutoProcessor.from_pretrained("Laskari-Naveen/ADA_II_Oct")
        model = VisionEncoderDecoderModel.from_pretrained("Laskari-Naveen/ADA_II_Oct")
        model.eval().to(device)
        print("Model loaded successfully")
    except:
        print("Model Loading failed !!!")
    return processor, model

import math
from collections import defaultdict

def calculate_key_aggregated_scores(scores, outputs, processor):
    """
    Calculate aggregated scores for each key from model outputs and scores.

    Args:
        scores (list): The list of score tensors for each decoding step.
        outputs (obj): The output object from the model containing generated sequences.
        processor (obj): The processor object for decoding tokens.

    Returns:
        dict: A dictionary with keys as the tokenized keys and their aggregated scores.
    """
    key_aggregated_scores = defaultdict(float)

    # Token IDs generated by the model (excluding input tokens like <s>)
    generated_token_ids = outputs.sequences[0][1:]  # Exclude the first token (<s>)

    current_key = None  # Track the current key during decoding
    token_scores = []  # Temporary list to store scores of intermediate tokens
    row_scores = []  # Temporary list for scores within semicolon-separated rows

    for idx, (score, token_id) in enumerate(zip(scores, generated_token_ids)):
        # Decode the token
        token = processor.tokenizer.decode([token_id.item()], skip_special_tokens=False)

        # Detect the start of a new key
        if token.startswith("<s_") and not token.startswith("</"):

            # print(f"Start of the token {token}")
            # Start a new key; reset token scores.
            # From a text remove <s_ and >
            current_key = token[3:-1]
            # current_key = token
            token_scores = []
            row_scores = []

        # Detect the end of the current key
        elif token.startswith("</") and current_key is not None:

            # print(f"End of the token {token}")
            # Compute the aggregated score for the key
            if token_scores:
                product_of_scores = math.prod(token_scores)
                aggregated_score = product_of_scores ** (1 / len(token_scores))
                row_scores.append(aggregated_score)

            # Assign row scores to the key
            key_aggregated_scores[current_key] = row_scores if len(row_scores) > 1 else row_scores[0]
            current_key = None  # Reset the key tracking
        
        # Process intermediate tokens
        elif current_key is not None:
            # Calculate the token's probability
            max_score = torch.softmax(score, dim=-1).max().item()

            # Handle row separators
            if token == ";":
                # print("Calculating Intermedeate")
                # Calculate and store the score for the current row
                if token_scores:
                    product_of_scores = math.prod(token_scores)
                    aggregated_score = product_of_scores ** (1 / len(token_scores))
                    row_scores.append(aggregated_score)
                    token_scores = []  # Reset token scores for the next row
            elif not token.startswith("<") and not token.startswith("</"):
                # Include the score for intermediate tokens
                # print(processor.tokenizer.decode([token_id.item()], skip_special_tokens=False))
                # print(max_score)
                token_scores.append(max_score)

        # elif current_key is not None:
        #     # TODO -: This might change if the number if beams change
        #     max_score = torch.softmax(score, dim=-1).max().item()
        #     # Include the score for intermediate tokens only
        #     if not token.startswith("<") and not token.startswith("</"):
        #         # Checking for which tokens the score is being included
        #         print(processor.tokenizer.decode([token_id.item()], skip_special_tokens=False))
        #         print(max_score)
        #         token_scores.append(max_score)

    return key_aggregated_scores


def convert_predictions_to_df(prediction):
    expanded_df = pd.DataFrame()
    result_df_each_image = pd.DataFrame()    
    each_image_output = pd.DataFrame(list(prediction.items()), columns=["Key", "Value"])
    # each_image_output["confidence_score"] = each_image_output["Key"].map(key_aggregated_scores)

    # print("each_image_output --->>>", prediction)
    # print("each_image_output isna --->>>", each_image_output[each_image_output["Value"].isna()])
    # print("empty_string_rows --->>>>", each_image_output[each_image_output["Value"] == ""])

    try:
        expanded_df = pd.DataFrame(columns=["Key", "Value"])
        for index, row in each_image_output[each_image_output["Value"].str.contains(";")].iterrows():
            expanded_rows = pd.DataFrame(split_and_expand(row))  # Expand rows
            # expanded_rows["confidence_score"] = key_aggregated_scores[row["Key"]]  # Assign the same score sum
            expanded_df = pd.concat([expanded_df, expanded_rows], ignore_index=True)

        result_df_each_image = pd.concat([each_image_output, expanded_df], ignore_index=True)
        result_df_each_image = result_df_each_image.drop(result_df_each_image[result_df_each_image['Value'].str.contains(';')].index)
        result_df_each_image = result_df_each_image.replace("<one>", "1")
    except Exception as e:
        print("Error in convert_predictions_to_df--->>>>", e)
        raise e
    return result_df_each_image

def map_result1(dict1, dict2):
    result_dict_1 = {}
    for key, value in dict1.items():
        if key in dict2:
            mapping_keys = dict2[key] if isinstance(dict2[key], list) else [dict2[key]]
            for mapping_key in mapping_keys:
                result_dict_1[mapping_key] = value
    return result_dict_1

def map_result2(dict1, dict2):
    result_dict_2 = {}
    for key, value in dict1.items():
        if key in dict2:
            mapping_keys = dict2[key] if isinstance(dict2[key], list) else [dict2[key]]
            for mapping_key in mapping_keys:
                result_dict_2[key] = {
                    "Mapping_key": mapping_keys,
                    "coordinates": value
                }
    return result_dict_2

def map_result1_final_output(result_dict_1, additional_info_dict, key_aggregated_scores):
    updated_result_dict_1 = {}

    # Iterate over additional_info_dict
    for key, additional_info in additional_info_dict.items():
        # Check if the key exists in result_dict_1
        if key in result_dict_1:
            coordinates = result_dict_1[key]
        else:
            # If the key is missing in result_dict_1, set coordinates to None
            coordinates = None
        
        if key in key_aggregated_scores:
            confidence_score = key_aggregated_scores[key]
        else:
            confidence_score = None

        # Store the coordinates and additional_info in updated_result_dict_1
        updated_result_dict_1[key] = {
            "coordinates": coordinates, 
            "text": additional_info, 
            "confidence_score" : confidence_score
        }

    return updated_result_dict_1

# Load the models
processor, model = load_model(device)



def run_ada_pipeline(image_path: str, logger, formatter):
    try:
        # image_path = os.path.join(input_image_folder, each_image)
        image_reading = "Image_reading"
        formatter.start_timing(image_reading)
        # log_message(logger, "Image_reading Started", level="INFO")
        pil_image = Image.open(image_path).convert('RGB')
        # pil_image = Image.open(io.BytesIO(image_path)).convert('RGB')
        to_tensor = transforms.ToTensor()
        image = to_tensor(pil_image)
        im_read_time = formatter.stop_timing(image_reading)
        log_message(logger, "Image_reading Completed", level="INFO", elapsed_time=im_read_time)
        
        Data_extraction = "Data Extraction"
        formatter.start_timing(Data_extraction)
        prediction, output, scores = run_prediction_donut(pil_image, model, processor)

        # Calculate key aggregated scores
        key_aggregated_scores = calculate_key_aggregated_scores(scores, output, processor)
        data_extraction_time = formatter.stop_timing(Data_extraction)
        log_message(logger, "Data Extraction and score computation", level="INFO", elapsed_time=data_extraction_time)
        # print("key_aggregated_scores --->>>> ", key_aggregated_scores)
        
        ex_post_processing = "Extraction Post Processing"
        formatter.start_timing(ex_post_processing)
        donut_out = convert_predictions_to_df(prediction)
        
        # This is just converting the dataframe to dictionary
        json_data = donut_out.to_json(orient='records')
        data_list = json.loads(json_data)
        output_dict_donut = {}
        
        # print("data_list ---->>>>", data_list)

        # Iterate through the data_list
        for item in data_list:
            key = item['Key']
            value = item['Value']   
            # score = item['confidence_score']

            # Check if the key already exists in the output dictionary
            if key in output_dict_donut:
                # If the key exists, append the value to the list of dictionaries
                output_dict_donut[key].append({'value': value})
            else:
                # If the key doesn't exist, create a new list with the current value
                output_dict_donut[key] = [{'value': value}]

        print("Length of Keys being outputed", len(output_dict_donut.keys()))
        
        # This is just doing the ROI inference and converting DF to dict
        data_ex_pp_time = formatter.stop_timing(ex_post_processing)
        log_message(logger, "Data Extraction post-processing", level="INFO", elapsed_time=data_ex_pp_time)

        ROI_extraction = "ROI_extraction"
        formatter.start_timing(ROI_extraction)   
        res = roi_model_inference(image_path, image)
        df_dict = res.to_dict(orient='records')
        ROI_time = formatter.stop_timing(ROI_extraction)
        log_message(logger, "ROI_extraction Completed", level="INFO", elapsed_time=ROI_time)

        # Now we just want that the classname should be the key and the values are the coordinates so 
        # output_dict_det has the class_name : x0 x1 y0 y1
        mapping_roi_donut = "Mappping Data Extraction and ROI"
        formatter.start_timing(mapping_roi_donut)   
        output_dict_det = {}
        for item in df_dict:
            class_name = item['class_name']
            x1, y1, x2, y2 = item['x0'], item['y0'], item['x1'], item['y1']
            output_dict_det[class_name] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}


        # Map the ROI keys with the Donut keys
        result_dict_1 = map_result1(output_dict_det, BBOX_DONUT_Mapping_Dict)
        # result_dict_2 = map_result2(output_dict_det, BBOX_DONUT_Mapping_Dict)
        final_mapping_dict  = map_result1_final_output(result_dict_1, output_dict_donut, key_aggregated_scores)
        
        Data_ex_roi_time = formatter.stop_timing(mapping_roi_donut)
        log_message(logger, "Mapping extracted data and ROI", level="INFO", elapsed_time=Data_ex_roi_time)

        return {"result": final_mapping_dict}, None
    except Exception as e:
        return None, str(e)

# with open('notes.json') as f:
#     category_mapping = {c['id'] + 1: c['name'] for c in json.load(f)['categories']}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your application description")
    parser.add_argument("input_image_folder", help="Path to the input image folder")
    parser.add_argument("output_ROI_folder", help="Path to the output ROI folder")
    parser.add_argument("output_extraction_folder", help="Path to the output extraction folder")
    args = parser.parse_args()
    processor, model = load_model(device)
    # run_application(args.input_image_folder, args.output_ROI_folder, args.output_extraction_folder)
