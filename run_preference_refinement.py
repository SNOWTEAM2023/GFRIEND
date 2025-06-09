import os
import json
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.dataset import PreferenceDataset
from generate.cot_sampling import CoTSampler
from generate.perplexity_scoring import PPLScorer

def run_preference_refinement(args):
    # 1. Build the data loader
    #    Here we only need (question, a-, a+) for loop calls
    dummy_tokenizer = None  # If only reading raw data, actual tokenization is not needed
    pref_dataset = PreferenceDataset(args.pref_data, dummy_tokenizer)

    # 2. Initialize sampler & scorer
    sampler = CoTSampler(args.sft_model_path, max_new_tokens=args.max_new_tokens)
    scorer = PPLScorer(args.sft_model_path)

    refined_data = []
    for i in tqdm(range(len(pref_dataset))):
        q, a_neg, a_pos = pref_dataset[i]
        # Treat (a-) as "Answer A" and (a+) as "Answer B" for example purposes
        generations = sampler.generate_judgment(
            question=q, 
            ansA=a_neg, 
            ansB=a_pos, 
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p
        )

        # Iterate through all generations
        for gen_str in generations:
            cot, chosen = scorer.parse_generation(gen_str)
            # Calculate the perplexity of the CoT
            ppl = scorer.compute_perplexity(cot)
            ppl_score = scorer.normalize_score(ppl, tau=args.tau)
            # Determine if the judgment is correct
            # Since a_pos is the better option in the original data, if chosen == "B", it is considered correct
            # (Assuming chosen="A" means pick a_neg, chosen="B" means pick a_pos)
            is_correct = (chosen.strip().upper() == "B")
            
            refined_data.append({
                "question": q,
                "answer_neg": a_neg,
                "answer_pos": a_pos,
                "cot": cot,
                "chosen": chosen,
                "is_correct": is_correct,
                "ppl": ppl,
                "ppl_score": ppl_score
            })

    # Classify based on ppl_score and is_correct: strong/weak accept/reject
    # The approach in the paper: Set a threshold p, e.g., 0.5
    #   ppl_score > p & is_correct=True => Strong Accept
    #   ppl_score <= p & is_correct=True => Weak Accept
    #   ppl_score > p & is_correct=False => Strong Reject
    #   ppl_score <= p & is_correct=False => Weak Reject
    # And perform Cartesian product of positive and negative samples to generate diverse (a+, a-) pairs
    threshold = args.threshold
    strong_accept = []
    weak_accept = []
    strong_reject = []
    weak_reject = []

    for item in refined_data:
        score = item['ppl_score']
        correct = item['is_correct']
        if score > threshold and correct:
            strong_accept.append(item)
        elif score <= threshold and correct:
            weak_accept.append(item)
        elif score > threshold and (not correct):
            strong_reject.append(item)
        else:
            weak_reject.append(item)

    # Perform Cartesian product of positive and negative groups to form new multi-level preference data
    # e.g., Each strong_accept combined with strong_reject => strong preference difference
    multi_pref_data = []
    def label_name(item):
        # For record-keeping purposes
        score = item['ppl_score']
        return ("StrongAccept" if score > threshold else "WeakAccept") if item['is_correct'] else \
               ("StrongReject" if score > threshold else "WeakReject")

    # Simple example: Combine SA/WA together and SR/WR together, then perform pairwise combinations
    import itertools
    positive_group = strong_accept + weak_accept
    negative_group = strong_reject + weak_reject

    for pos_item, neg_item in itertools.product(positive_group, negative_group):
        # Both pos_item and neg_item contain the question
        # Their chosen answers are in 'chosen'
        # The constructed pair indicates "pos_item's judgment > neg_item's judgment"
        multi_pref_data.append({
            "question": pos_item['question'],
            "pos_cot": pos_item['cot'],
            "neg_cot": neg_item['cot'],
            "pos_ppl_score": pos_item['ppl_score'],
            "neg_ppl_score": neg_item['ppl_score'],
            "pos_label": label_name(pos_item),
            "neg_label": label_name(neg_item),
            # Preserve original optional information
            "pos_answer_pos": pos_item['answer_pos'],
            "pos_answer_neg": pos_item['answer_neg'],
            "neg_answer_pos": neg_item['answer_pos'],
            "neg_answer_neg": neg_item['answer_neg'],
        })

    # Output multi-level preference data
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "multi_pref_data.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        for line in multi_pref_data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(f"[INFO] Multi-level preference data saved to {out_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_model_path", type=str, required=True,
                        help="Path to the SFT-finetuned model.")
    parser.add_argument("--pref_data", type=str, required=True,
                        help="Path to the initial preference data (q, a-, a+).")
    parser.add_argument("--output_dir", type=str, default="./refined_outputs")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    run_preference_refinement(args)