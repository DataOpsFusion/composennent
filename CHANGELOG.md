# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.6] - 2024

### Added
- Unified training interface with `.pretrain()` and `.fine_tune()` methods on models
- Instruction fine-tuning support (Alpaca format)
- Multiple trainer classes: CausalLMTrainer, MaskedLMTrainer, Seq2SeqTrainer, MultiTaskTrainer
- Custom loss function support via CustomTrainer
- Automatic mixed precision (AMP) training

### Changed
- Improved API consistency across modules

## [0.3.0] - 2024

### Added
- Initial public release
- Core transformer components (Encoder, Decoder, Block)
- GPT and BERT model implementations
- WordPiece and SentencePiece tokenizer support
- Vision transformer components
- Mixture of Experts (MoE) components
- Training utilities and dataloaders
