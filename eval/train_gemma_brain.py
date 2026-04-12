#!/usr/bin/env python3
"""Train Gemma 4 E2B on brain graph data — three format variations.

Upload this + the three JSONL files to Google Colab (free T4 GPU).
Trains three LoRA adapters, one per traversal format.

Usage (in Colab):
    !pip install unsloth
    !python train_gemma_brain.py --format B --data B_typed_walks.jsonl
    !python train_gemma_brain.py --format C --data C_topdown.jsonl
    !python train_gemma_brain.py --format C2 --data C2_bottomup.jsonl

Each run: ~5-10 minutes on free T4. Produces a LoRA adapter (~16MB).
"""

import argparse
import json
import os


def load_dataset(path):
    """Load JSONL training data into HuggingFace dataset format."""
    from datasets import Dataset

    texts = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            texts.append(ex['text'])

    return Dataset.from_dict({"text": texts})


def train(format_name, data_path, output_dir=None):
    """Train Gemma 4 E2B with QLoRA on one format."""
    from unsloth import FastModel
    from trl import SFTTrainer, SFTConfig

    if output_dir is None:
        output_dir = "brain-adapter-%s" % format_name

    print("=" * 60)
    print("Training brain model: Format %s" % format_name)
    print("Data: %s" % data_path)
    print("Output: %s" % output_dir)
    print("=" * 60)

    # Load model — 4-bit quantized, ~2GB
    print("\nLoading Gemma 4 E2B...")
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-4-E2B-it",
        max_seq_length=4096,  # our examples are ~1K tokens, 4K is plenty
        load_in_4bit=True,
        full_finetuning=False,
    )

    # Apply LoRA
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,          # rank — higher = more capacity, more memory
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )

    # Load training data
    print("\nLoading dataset...")
    dataset = load_dataset(data_path)
    print("  %d examples" % len(dataset))

    # Train
    print("\nTraining...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=3,          # 3 passes over data
            learning_rate=2e-4,
            optim="adamw_8bit",
            logging_steps=10,
            output_dir=output_dir,
            save_strategy="no",          # save only at end
            report_to="none",
            fp16=not os.environ.get("COLAB_TPU_ADDR"),
            bf16=bool(os.environ.get("COLAB_TPU_ADDR")),
        ),
    )

    stats = trainer.train()
    print("\nTraining complete!")
    print("  Loss: %.4f" % stats.training_loss)
    print("  Steps: %d" % stats.global_step)

    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("  Adapter saved to: %s" % output_dir)

    return stats


def test_model(adapter_dir, test_queries=None):
    """Quick test — ask the trained model about the brain."""
    from unsloth import FastModel

    if test_queries is None:
        test_queries = [
            "What do you know about boot architecture?",
            "What corrections were made to the encoding agent?",
            "How does the fractal architecture work?",
            "What did Tom say about testing?",
            "What is the relationship between recall and encoding?",
        ]

    print("\n" + "=" * 60)
    print("Testing adapter: %s" % adapter_dir)
    print("=" * 60)

    model, tokenizer = FastModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastModel.for_inference(model)

    for query in test_queries:
        print("\nQ: %s" % query)
        messages = [{"role": "user", "content": query}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt").to(model.device)

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0][inputs.shape[-1]:],
                                     skip_special_tokens=True)
        print("A: %s" % response[:500])
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", required=True, choices=["B", "C", "C2"],
                        help="Training format: B (typed walks), C (top-down), C2 (bottom-up)")
    parser.add_argument("--data", required=True, help="Path to JSONL training file")
    parser.add_argument("--output", help="Output directory for adapter")
    parser.add_argument("--test", action="store_true", help="Run test queries after training")
    parser.add_argument("--test-only", help="Test an existing adapter (skip training)")
    args = parser.parse_args()

    if args.test_only:
        test_model(args.test_only)
    else:
        train(args.format, args.data, args.output)
        if args.test:
            output = args.output or ("brain-adapter-%s" % args.format)
            test_model(output)
