This is the pipeline for pre-processing the RSC 6.0.4 OPEN
1. Original data: https://fedora.clarin-d.uni-saarland.de/rsc_v6/access.html#download
- We would use the raw data (.txt), then use Stanza to split the sentences, POS and depedency parsed the files

2. Pipeline
2.1. Get the years of the file and append it to the end of the file name
2.2. Parse using stanza
2.3. TBA