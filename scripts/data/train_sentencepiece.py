"""Train the SpikeRuGPT SentencePiece tokenizer from a prepared text sample."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/tokenizer_sample/spikerugpt_tokenizer_sample.txt")
    parser.add_argument("--model-prefix", default="tokenizer/spikerugpt-bpe-32k")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--model-type", choices=["bpe", "unigram"], default="bpe")
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    parser.add_argument("--input-sentence-size", type=int, default=5000000)
    parser.add_argument("--no-shuffle-input-sentence", action="store_true")
    parser.add_argument("--no-byte-fallback", action="store_true")
    parser.add_argument("--num-threads", type=int, default=16)
    args = parser.parse_args()

    import sentencepiece as spm

    Path(args.model_prefix).parent.mkdir(parents=True, exist_ok=True)
    user_symbols = [
        "<|endoftext|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
    ]
    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=not args.no_shuffle_input_sentence,
        num_threads=args.num_threads,
        byte_fallback=not args.no_byte_fallback,
        hard_vocab_limit=False,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=user_symbols,
    )
    print(f"Wrote {args.model_prefix}.model and {args.model_prefix}.vocab", flush=True)


if __name__ == "__main__":
    main()
