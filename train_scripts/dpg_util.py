import torch
from PIL import Image
import os.path as osp
import os
class MPLUG(torch.nn.Module):
    def __init__(self, ckpt='damo/mplug_visual-question-answering_coco_large_en', device='gpu'):
        super().__init__()
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        if os.environ.get('MPLUG_PATH'):
            ckpt = os.environ.get('MPLUG_PATH')
        self.pipeline_vqa = pipeline(Tasks.visual_question_answering, model=ckpt, device=device)

    def vqa(self, image, question):
        input_vqa = {'image': image, 'question': question}
        result = self.pipeline_vqa(input_vqa)
        return result['text']

def crop_image(input_image, crop_tuple=None):
    if crop_tuple is None:
        return input_image

    cropped_image = input_image.crop((crop_tuple[0], crop_tuple[1], crop_tuple[2], crop_tuple[3]))

    return cropped_image


def compute_dpg_one_sample(question_dict, image_path, vqa_model, resolution):
    generated_image = Image.open(image_path)
    crop_tuples_list = [
        (0,0,resolution,resolution),
        (resolution, 0, resolution*2, resolution),
        (0, resolution, resolution, resolution*2),
        (resolution, resolution, resolution*2, resolution*2),
    ]
    pic_num = 1
    crop_tuples = crop_tuples_list[:pic_num]
    #key = osp.basename(image_path).split('.')[0]
    value = question_dict
    qid2tuple = value['qid2tuple']
    qid2question = value['qid2question']
    qid2dependency = value['qid2dependency']

    qid2answer = dict()
    qid2scores = dict()
    qid2validity = dict()

    scores = []
    for crop_tuple in crop_tuples:
        cropped_image = crop_image(generated_image, crop_tuple)
        for id, question in qid2question.items():
            answer = vqa_model.vqa(cropped_image, question)
            qid2answer[id] = answer
            qid2scores[id] = float(answer == 'yes')
            with open(image_path+ '.detail.txt', 'a') as f:
                f.write(image_path + ', ' + str(crop_tuple) + ', ' + question + ', ' + answer + '\n')
        qid2scores_orig = qid2scores.copy()

        for id, parent_ids in qid2dependency.items():
            # zero-out scores if parent questions are answered 'no'
            any_parent_answered_no = False
            for parent_id in parent_ids:
                if parent_id == 0:
                    continue
                if qid2scores[parent_id] == 0:
                    any_parent_answered_no = True
                    break
            if any_parent_answered_no:
                qid2scores[id] = 0
                qid2validity[id] = False
            else:
                qid2validity[id] = True

        score = sum(qid2scores.values()) / len(qid2scores)
        scores.append(score)
    average_score = sum(scores) / len(scores)
    with open(image_path+'.summary.txt', 'a') as f:
        f.write(image_path + ', ' + ', '.join(str(i) for i in scores) + ', ' + str(average_score) + '\n')
  
    return average_score, qid2tuple, qid2scores_orig

class DPGFeedback:
    
    def __init__(self,device):
        self.vqa_model = MPLUG(device=device)
        
    def evaluate_image(self,image_path,metadata):
        question_dict = metadata['vqa_data']
        resolution = 1024
        score, _, _ = compute_dpg_one_sample(question_dict, image_path, self.vqa_model, resolution)
        return dict(
            correct=score,
            text_feedback='',
            prompt=metadata['prompt'],
            prompt_id=metadata['key'],
        )
from collections import defaultdict
import pandas as pd
def prepare_dpg_data(csv_file,prompt_path):
    previous_id = ''
    current_id = ''
    question_dict = dict()
    category_count = defaultdict(int)
    data = pd.read_csv(csv_file)
    for i, line in data.iterrows():
        if i == 0:
            continue

        current_id = line.item_id
        qid = int(line.proposition_id)
        dependency_list_str = line.dependency.split(',')
        dependency_list_int = []
        for d in dependency_list_str:
            d_int = int(d.strip())
            dependency_list_int.append(d_int)

        if current_id == previous_id:
            question_dict[current_id]['qid2tuple'][qid] = line.tuple
            question_dict[current_id]['qid2dependency'][qid] = dependency_list_int
            question_dict[current_id]['qid2question'][qid] = line.question_natural_language
        else:
            question_dict[current_id] = dict(
                qid2tuple={qid: line.tuple},
                qid2dependency={qid: dependency_list_int},
                qid2question={qid: line.question_natural_language})
        
        category = line.question_natural_language.split('(')[0].strip()
        category_count[category] += 1
        
        previous_id = current_id

    final_data  =[]
    for key in question_dict.keys():
        tgt = osp.join(prompt_path,key+'.txt')
        with open(tgt) as f:
            prompt = f.read()
        prompt = prompt.strip()
        final_data.append(
            dict(vqa_data=question_dict[key],
            key=key,
            prompt=prompt)
        )
    return final_data
