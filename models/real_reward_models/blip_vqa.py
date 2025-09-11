import torch
from lavis.models import load_model_and_preprocess
from lavis.models.base_model import tile
import torch.nn.functional as F
from PIL import Image

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
        
        # # Freeze everything
        # for param in self.blip_model.parameters():
        #     param.requires_grad = False

        # # Unfreeze only the linear decoder (prediction layer)
        # for param in self.blip_model.text_decoder.cls.predictions.decoder.parameters():
        #     param.requires_grad = True

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


# ---------------------
# Example usage
# ---------------------
image = Image.open("sdxl/1/0.png")
question = "What is happening on the branch where two birds are perched, one chirping happily and the other listening silently?"

vqa_model = VQAModel('cuda')
score = vqa_model.get_score(image, question)
print(score)
