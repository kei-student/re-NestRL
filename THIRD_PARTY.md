The optional `external/PACE` directory contains official upstream PACE sources
from <https://github.com/bigai-ai/PACE>, commit
`f008c19aa743ec209550375c9ee9c25379ac8b24`. It is downloaded by the bootstrap
script, excluded from version control, and remains unmodified. Its license and
attribution files remain in the downloaded source. The reconstruction executes
six cooking world / recipe modules; the file hashes are recorded in
`scripts/upstream_files.json`.

The locally provided `nestRL.pdf` is the user-supplied reference and is excluded
from version control. The NestRL paper is available at
<https://arxiv.org/html/2602.17737v2> under the license stated by its authors.

No earlier local NestRL reproduction code or checkpoints are imported.
