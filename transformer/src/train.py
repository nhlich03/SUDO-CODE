import os
from utils import load_config, ensure_dir
from dataset import load_and_split
from model import TranslationModel
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate


def preprocess_function(tokenizer, examples, src_col, tgt_col, max_src_len, max_tgt_len):
    model_inputs = tokenizer(examples[src_col], max_length=max_src_len, truncation=True, padding="max_length")
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples[tgt_col], max_length=max_tgt_len, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    ensure_dir(cfg["train"]["output_dir"])

    # Load data
    train_ds, val_ds, test_ds = load_and_split(
        cfg["data"]["path"],
        src_col=cfg["data"]["source_col"],
        tgt_col=cfg["data"]["target_col"],
        test_size=cfg["data"]["test_size"],
        val_ratio_of_test=cfg["data"]["val_ratio_of_test"],
        seed=cfg.get("seed", 42),
    )

    # Load model & tokenizer
    tm = TranslationModel(cfg["model_name"])
    tokenizer = tm.tokenizer
    model = tm.model

    # Preprocess (tokenize)
    tokenized_train = train_ds.map(lambda ex: preprocess_function(tokenizer, ex, cfg["data"]["source_col"], cfg["data"]["target_col"], cfg["train"]["max_source_length"], cfg["train"]["max_target_length"]), batched=True)
    tokenized_val = val_ds.map(lambda ex: preprocess_function(tokenizer, ex, cfg["data"]["source_col"], cfg["data"]["target_col"], cfg["train"]["max_source_length"], cfg["train"]["max_target_length"]), batched=True)

    # Set format
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Metrics
    metric = evaluate.load("sacrebleu")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [[l] for l in decoded_labels]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    # Training args
    targs = Seq2SeqTrainingArguments(
        output_dir=cfg["train"]["output_dir"],
        evaluation_strategy=cfg["train"].get("evaluation_strategy", "epoch"),
        learning_rate=cfg["train"].get("learning_rate", 5e-5),
        per_device_train_batch_size=cfg["train"].get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=cfg["train"].get("per_device_eval_batch_size", 16),
        weight_decay=cfg["train"].get("weight_decay", 0.01),
        num_train_epochs=cfg["train"].get("num_train_epochs", 5),
        predict_with_generate=True,
        logging_steps=cfg["train"].get("logging_steps", 50),
        disable_tqdm=False,
        report_to="none",
        push_to_hub=False,
        save_total_limit=cfg["train"].get("save_total_limit", 2),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(cfg["train"]["output_dir"])
    tokenizer.save_pretrained(cfg["train"]["output_dir"])


if __name__ == "__main__":
    main()
