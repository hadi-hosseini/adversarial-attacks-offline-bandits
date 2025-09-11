import torch
from lavis.models import load_model_and_preprocess
from lavis.models.base_model import tile
import torch.nn.functional as F
from PIL import Image

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.reward_architecture import fw0_and_grad

# def predict_answer(samples,  answer_list, num_ans_candidates=2):
#     if isinstance(samples["text_input"], str):
#         samples["text_input"] = [samples["text_input"]]
#     num_ans_candidates = min(num_ans_candidates, len(answer_list))
#     return custom_rank_answers(samples, answer_list, num_ans_candidates)


def custom_rank_answers(self, samples, answer_list, num_ans_candidates):
    answer_candidates = self.tokenizer(
        answer_list, padding="longest", return_tensors="pt"
    ).to(self.device)
    answer_candidates.input_ids[:, 0] = self.tokenizer.bos_token_id

    answer_ids = answer_candidates.input_ids
    answer_atts = answer_candidates.attention_mask

    question_output, _ = self.forward_encoder(samples)
    question_states = question_output.last_hidden_state

    tokenized_question = samples["tokenized_text"]
    question_atts = tokenized_question.attention_mask

    num_ques = question_states.size(0)
    start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

    start_output = self.text_decoder(
        start_ids,
        encoder_hidden_states=question_states,
        encoder_attention_mask=question_atts,
        return_dict=True,
        reduction="none",
    )
    logits = start_output.logits[:, 0, :]  # first token's logit

    answer_first_token = answer_ids[:, 1]
    prob_first_token = F.softmax(logits, dim=1).index_select(
        dim=1, index=answer_first_token
    )
    topk_probs, topk_ids = prob_first_token.topk(num_ans_candidates, dim=1)

    input_ids = []
    input_atts = []
    for b, topk_id in enumerate(topk_ids):
        input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
        input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
    input_ids = torch.cat(input_ids, dim=0)
    input_atts = torch.cat(input_atts, dim=0)

    targets_ids = input_ids.masked_fill(
        input_ids == self.tokenizer.pad_token_id, -100
    )

    question_states = tile(question_states, 0, num_ans_candidates)
    question_atts = tile(question_atts, 0, num_ans_candidates)

    output = self.text_decoder(
        input_ids,
        attention_mask=input_atts,
        encoder_hidden_states=question_states,
        encoder_attention_mask=question_atts,
        labels=targets_ids,
        return_dict=True,
        reduction="none",
    )

    log_probs_sum = -output.loss
    log_probs_sum = log_probs_sum.view(num_ques, num_ans_candidates)

    max_topk_ids = log_probs_sum.argmax(dim=1)
    max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]
    answers = [answer_list[max_id] for max_id in max_ids]
    topk_probs_ = topk_probs.detach().cpu().numpy()
    probs = [(topk_probs_[i,0],topk_probs_[i,1]) if max_id==0 else (topk_probs_[i,1],topk_probs_[i,0]) for i,max_id in enumerate(max_ids)]
    return answers, probs


class VQAModel:
    def __init__(self, device='cuda'):
        self.device = device
        self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name="blip_vqa", model_type="vqav2", is_eval=True, device=device
        )
        self.blip_model._rank_answers = custom_rank_answers.__get__(self.blip_model, type(self.blip_model))

        self.cls =  self.blip_model.text_decoder.cls

        for param in self.blip_model.parameters():
            param.requires_grad = False

        for param in self.blip_model.text_decoder.cls.parameters():
            param.requires_grad = True
        

    def get_score(self, image, question):
        image_ = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        question_ = self.txt_processors["eval"](question)
        
        with torch.no_grad():
            vqa_pred = self.blip_model.predict_answers(
                samples={"image": image_, "text_input": question_}, 
                inference_method="rank", 
                answer_list=['yes','no'],
                num_ans_candidates=2
            )
        pos_score, _ = vqa_pred[1][0][0], vqa_pred[1][0][1]
        return pos_score
    

    def get_cls_input(self, image, question):
        """Return the hidden states that go into the CLS head"""
        image_ = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        question_ = self.txt_processors["eval"](question)

        samples = {"image": image_, "text_input": question_}
        question_output, _ = self.blip_model.forward_encoder(samples)
        question_states = question_output.last_hidden_state

        tokenized_question = samples["text_input"]
        # Ensure it's tokenized properly for the decoder
        if isinstance(tokenized_question, str):
            tokenized_question = self.blip_model.tokenizer(tokenized_question, return_tensors="pt").to(self.device)

        # Pass through text decoder BERT
        decoder_output = self.blip_model.text_decoder.bert(
            input_ids=tokenized_question.input_ids,
            attention_mask=tokenized_question.attention_mask,
            encoder_hidden_states=question_states,
            encoder_attention_mask=None,
            return_dict=True
        )
        hidden_states = decoder_output.last_hidden_state  # <-- input to cls head
        return hidden_states



image = Image.open("data/generative_models/sdxl/1/0.png")
question = "Are there birds in the image?"

vqa_model = VQAModel('cuda')
score = vqa_model.get_score(image, question)
print(score)
hidden_state = vqa_model.get_cls_input(image, question)

f_w0 = vqa_model.cls(hidden_state).squeeze()



exit()
params = [p for p in vqa_model.cls.parameters() if p.requires_grad]
grads = torch.autograd.grad(f_w0[0,0,yes_token_id], params, retain_graph=False, create_graph=False)
grad_flat = torch.cat([g.reshape(-1) for g in grads])


# f_x, grad_x = fw0_and_grad(vqa_model.cls, torch.tensor(hidden_state, dtype=torch.float32, device='cuda'))
# print(grad_x)
# print(f_x)
# print(score)

