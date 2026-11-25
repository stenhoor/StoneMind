# Work in progress: spike 0001, phase 2

This package is incomplete and is not yet a workspace. It holds:

- `src/model/`: the KataGo `.bin.gz` parser (written, untested).
- `src/reference/`: the fp32 reference forward pass and its post-processing (written, untested).
- `src/features/simple.ts`: the V7 input encoder for simple positions (written, untested).

Still to come before its PR: package.json and tsconfig, symmetry helpers, tests against KataGo's tiny-board reference
output and the g170-b6c96 test network (fetched by pinned commit and SHA-256), and the script that captures
golden outputs from native KataGo.
