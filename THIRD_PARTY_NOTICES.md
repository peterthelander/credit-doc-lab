# Third-party notices

## ExtractBench (Contextual AI)

The `prepare` command downloads the credit-agreement subset of
[ContextualAI/extract-bench](https://github.com/ContextualAI/extract-bench) (pinned commit
`ca817638852a23bb17ca9aaeb82ccdd29e9af9ca`): source PDFs, the extraction schema, and
human-validated reference answers. Downloaded files are not committed here; they are placed
under `data/` (git-ignored) together with `LICENSE.extract-bench` and `ATTRIBUTION.txt`.
The optional `official` extra installs the benchmark's evaluation package from the same commit.

Citation requested by the authors:

```bibtex
@article{ferguson2026extractbench,
  title={ExtractBench: A Benchmark and Evaluation Methodology for Complex Structured Extraction},
  author={Ferguson, Nick and Pennington, Josh and Beghian, Narek and Mohan, Aravind and Kiela, Douwe and Agrawal, Sheshansh and Nguyen, Thien Hang},
  journal={arXiv preprint arXiv:2602.12247},
  year={2026}
}
```

The repository's license is MIT, Copyright (c) 2026 Contextual AI. The MIT license requires this
copyright and permission notice to be included in copies or substantial portions of the software;
the full text is saved as `data/credit_agreement/LICENSE.extract-bench` by `prepare`
and at <https://github.com/ContextualAI/extract-bench/blob/ca817638852a23bb17ca9aaeb82ccdd29e9af9ca/LICENSE>.

The dataset README only says "See the repository LICENSE for terms." The paper (Appendix A) says
the credit agreements are public-domain exhibits from SEC EDGAR filings and that the dataset and
framework are released under the MIT license. That is the authors' statement; it has not been
independently verified. Check before redistributing the downloaded data.
