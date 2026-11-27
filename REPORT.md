# Report

## The high-level breakdown

1. I chose the Hugging Face Open ASR Leaderboard, the headlining number besides private data, on the top five open-weight models. No subset was chosen, except for private due to not having access to the reference data. There was no real difference in the published WER except for Nemotron, which seems to have a mismatch between uploaded transcriptions and reported figures.
2. I split the leaderboard normalization into each step and added four more. The largest contributor by far is punctuation and capitalization. Please refer to the `Normalization` section for more details.
3. I added semantic error, which embeds the reference and hypotheses with a sentence embedder and compares the two vectors via cosine similarity. This informs targets for agentic work. Please refer to the `Metrics` section for more info.
4. I gathered the given (by the benchmark) metadata and added acoustic metadata. I tested and made a report of reliability of the tests, with most being reliable except for estimated SNR degrading with more audio degradations. Please refer to the `Metadata` section for more information.

Please refer to the sections `Limitations`, `Automation`, and `AI usage` for their breakdowns.

## Reproducing the Hugging Face ASR benchmarks

The dataset that I chose to reproduce is the Hugging Face Open ASR Leaderboard, and in particular the headlining number. I used the top five open-weight models and the open benchmarks (not private) because of access to their transcriptions in the Hugging Face bucket, which makes reproduction easy for anyone to copy and emulate.

I chose the entirely of these benchmarks, so no subset was chosen, and it is as large as what they include (except for the private, which by it's very nature I do not have).

I was able to recreate it for all the models except for Nemotron, which seems to publish different figures than what is on the main page. From AI research: "The public files were uploaded on 14 November, four days before the leaderboard's code gained support for this build, so those seven figures may be from a later run whose transcripts were not published."

## Normalization

I tested out several normalizers to see the difference in how all of them fared:

- Verbatim: The text as received out of the ASR system with no change as a control
- Case-punct: Only capitalization and punctuation (other than apostrophes) are removed
- Whisper-stock: The stock Whisper normalizer
- Leaderboard: The Hugging Face Open ASR Leaderboard normalizer, as realized in the repo
- Leaderboard_split: The normalizer split into parts to see how each contributed
- Own: Four additional normalizations added on top of the leaderboard

By looking at ablate or the heatmap in the web UI, one can see that punctuation, case, and hyphens have the biggest impact on the error rate, for all metrics. This makes sense to do and is an easy win, but also shows the messy nature of the open benchmarks and the output from the open-weight models.

Fillers also make a big impact, but show a mismatch between these benchmarks and the expected performance from an ASR system (as opposed to STT with smart formatting). Stripping them removes information and now relies on a hand-curated list of fillers to remove to get a good estimate of a system (if you even want this behavior all the time). Consistency is key.

Numerals are the same as fillers except that information is also lost. The term 1204 can be said a myriad of ways, but written in numerals only one. Information is lost from the benchmark if you test this way.

Compounds do not have an outsized influence, but are worth thinking about for downstream systems being able to ingest properly.

Regional spelling and alt spellings also don't have an outsized influence on the benchmarks, but this will have a larger effect when dealing with non-American spellings, since models will likely produce American spellings. In addition, it is problematic to curate a hand-made or crawled list of variations. Not only does it introduce the possibility of mistakes, but while it might not be an infinite list, it likely is ever growing. There is also the issue of spelling mistakes in the reference or even propagated to ASR systems leading to wrong normalization.

I added four from analysis of the actual benchmarks:

1. Cut-off words (single th- or w) at the ends of words. This doesn't have a huge effect but does lead to difficulties. Single vowels could be there too, but can be words and not unambiguously waved away.
2. Interjections: This is the same issue as fillers.
3. Discourse markers: This is the same issue as fillers too.
4. Repetitions: The use case of this was reference labels that dropped repeated words "i i", but the implementation leads to issues with initialisms. This doesn't make a large impact but it shows the need for proper labels.

## Metrics

### Measurements

The most common measurement for ASR systems is Word Error Rate (WER), which is how many words are wrong in a sentence, usually expressed as a percentage. This can go over 100% because one can insert a million words in a one-word sentence and then you are really, really wrong. This is generated from the insertion, deletion, and substitution errors.

Another measurement is CER, or character error rate. This is just over which characters are wrong, which can be useful for languages that don't have word-based orthography or when judging alphanumeric phrases.

The last measurement that I decided to add is semantic error rate. This measures how similar two phrases are to each other. "I sat" and "I ran" differ greatly, but "I sat" and "I took a seat" are more similar despite having a worse WER. This is generated by embedding the reference and hypotheses in a sentence embedder and then doing cosine similarity on their vectors. This is not perfect, as negation can have little impact on these differences despite having opposite meanings to a sentence, though some work has been done on negation-tuned embeddings. This measurement would be useful to inform agents where transcription is not shown to the user but just responses from the system. If the correct outcome resulted despite a higher WER, it gives information to the modeling team. It is a better target for those situations.

As a whole, the order of how bad queries score is WER, Semantic Error, and then CER.

### Aggregations

The most common way of displaying WER is by overall, or pooled, where all reference words are combined, all errors combined, and then computed. This minimizes the impact of short queries, where a single error is a large percentage, but masks the reality of some of the queries. By looking at median (over multiple bins) one can see that there are many perfectly 0% WER queries and the errors can cluster on the ones that are wrong. By looking at overall or even mean WER, one can be led to the belief that errors are distributed evenly across the data.

## Metadata

There are two sources of metadata in this study: those that are given and those that are derived. Given is usually trustworthy, but understanding how some of the references can have typos and issues, there always needs to be the understanding that these can be wrong sometimes. For this study, most of the derived metadata is focused on acoustic information. This was chosen because of my intuition that noise was a large factor in the issues.

For the description of this metadata, Earnings22 is removed from the discussion since audio files were long earnings calls and not derived from the individual clips. Most trends come from GigaSpeech because of its size (35 hours) compared to the other benchmarks, unless otherwise noted. This allows more statistical power for its results.

- **Duration**: How long the audio is in seconds. These tend to skew shorter, as is the classic marker of benchmarks and ASR systems. Long queries have a higher tendency to have issues, which could be because of compounding errors in the transcription or probability over the time of the query. There does seem to be a trend to 
- **Sample rate**: This tends to be even across benchmarks, with the exception of URGENT 2024, which had the particular quality of a larger sample rate being associated with a worse WER. After researching this, these tended to be a particular subset of queries from the same source, so sample rate may not affect anything but it might be the underlying conditions of those clips. In general, lower sample rate is associated with lower scores because more acoustic information is missing.

- **Ref words**: The number of words in the reference. Same trend as duration.

- **Alphanumeric**: The presence of numerals or alpha/numeral mixes. This doesn't show a strong difference between the two camps (present or not) but alphanumerics aren't well represented, which is not representative of many real-world situations.

- **Loudness**: How loud a query tends to be. This doesn't seem to have a strong signal either way.

- **SNR given and SNR estimated**: Signal to noise ratio, or how noisy something is. The lower the ratio, the more noise there is drowning out the signal. URGENT 2024 artificially augmented the dataset from a clean version into a noisy one. The measurement of given SNR to estimated SNR is different, but the two calculations do have issues. One doesn't account for quiet in the real clip and the other is imperfect by its measurement. It also has floor and ceiling issues. A full estimated SNR would need refinement. There does seem to be a correlation between lower SNR and increased error, for both given and estimated values.

- **Bandwidth**: Where most of the frequency information is. This can be used to catch resampled data, if a 16 kHz audio only has the range of an 8 kHz audio. This does seem to correlate, with the lower the bandwidth the higher the error, which corresponds to acoustic signals missing.

- **Speech fraction**: How much of the query has speech present. This uses a voice activity model to classify frames into voice activity or not, which can give estimates on how much of the audio is spoken vs other. As more speech dominates, the WER does improve, which does indicate that these test cases are tailored exactly for a benchmark. Looking at the median shows that the lower the speech fraction the higher the error, which would indicate hallucination on noise other than the speech signal.

- **Speaking rate**: An estimate of how fast a speaker is talking. This is derived from the duration and ref words. Generally it seems that the faster one is talking, the higher the error.

### Testing the metadata

A good portion of the metadata was given, so it is not tested, like duration. However, utilizing the URGENT 2024 benchmark's clean and noisy benchmark, we can check the acoustic estimates that we got. Just some background on URGENT 2024: the noisy set was created from the clean by adding noise, cut off (sampled down and then back up), clipped (the top clipped off) or reverberated. We have a list of what items got what added to them from the benchmark's repo.

- **Loudness**: We can artificially raise the volume of a query and see if we measure an increased level for the changed test case.

- **SNR estimated**: We can correlate the measured SNR against what is given. We can also run some tests of our own with white noise and see if we can correlate it. This does a pretty good job of testing it.

- **Bandwidth**: We know which queries were cut off with resampling, but we don't know what frequency it was cut off at, so we instead resample and see how well we can pick up that banding difference.

- **Speech fraction**: We can check between noisy and clean test cases and see if the speech fraction is kept the same between the two. This doesn't inherently test the speech fraction, since it relies on a model, but it tests its noise robustness.

The full results can be seen if you run `uv run ba test_metadata`, but here is the short of it:

- SNR estimate is good when just noise is applied, but gets worse as other degradations get applied.
- Speech fraction also gets worse with more degradations.
- Bandwidth does accurately retrieve the information about resampling.
- Loudness is captured well.

### Binning

A note on binning for metadata slices: the number of bins should correspond to the size of the benchmark, otherwise noise will drive the analysis. Ten is a good number for the smaller, while higher can be used for GigaSpeech.

## Limitations

- These benchmarks are just in English, which leads to metadata not being relevant, like language. It also covers the most worked-on language so the models tend to cluster on fairly good scores.
- Reliant on transcriptions from the Hugging Face buckets. This makes sense for cost and the scope of this project, but as can be seen with Nemotron, that can lead to some issues and trust is involved.
- Reliant on open-weight models. This was for cost, complexity, and the legality of uploading benchmark results to GitHub, but there is a whole side of the market not viewed by this report/system right now.
- These benchmarks are not fully representative of real life. While each benchmark has a real use-case, like transcription of meetings or videos, the noisy real-world queries from real users out in the noisy world are not really represented.
- No medical or keyterm recall testing. This is a huge untouched aspect: accurately getting drug names or account numbers.
- I focused on acoustic conditions as my derived metadata, but there is a whole other wealth of information that can be gleaned from real production data, such as location, accent, device, known use case, etc.
- The types of errors are not represented in the project as is (mentioned in future work)

## Automation

These ideas go beyond just using AI to build this out and are more about the workflow automation.

- Automated grabbing of newest models and results from Hugging Face to keep up to date
- Feed data to AI from analysis of the metadata-labeled and scored data. This has extreme potential and even in this project I saw good use from using AI. You could make a change (say test a new normalization), run the scoring and ask AI to look at trends that changed. There are issues with size and context for the LLM, so some condensing and refinement need to happen if you have datasets larger than these benchmarks. When used for this project, one finding was that errors cluster around particular speakers/episodes/calls. This finding doesn't show what is going on with those calls, but leads the researcher to a beginning place to research what sets those queries apart.
- Add LLM metadata, like a classification for the type of utterance in the reference (along language-specific lines, like "question" or "request")

## AI usage

AI was used throughout this project, not just for building the product, but for analysis of the data as well.

I began by outlining many different avenues of research and then whittled it down to what would be the best. I used AI to do basic research on the benchmarks and feasibility of the experiments.

I outlined the specs of the app that I wanted to make with key decisions made beforehand regarding organization, goal, and some specific frameworks/tools.

From there, I let it build out the basic structure of the app and working together we iterated on the design. We ranged from high level to small implementations of things. A good portion of what I ended up doing was simplifying and reducing the complexity added with an eye towards maintainability and understanding. I had to keep an eye out for fake features, such as a "natural language parser" (unasked for, I must add), which was just a regex for certain words to convert to filtering. I had to rework a lot of the organization and structure to be able to be added to and reviewed easily. It did help with some implementations of acoustic measurements, but where I could I replaced those with outside open libraries that have been proven by use.

Fetching data from Hugging Face and the UI with Streamlit I left mostly alone since I could verify that by looking at the data, the reproduction of the WER, and by user testing of the web UI.

## Future work

- Clip grabber w/results to look at information at the particular query (though using AI directly on the data does well with the structure I have set up with the scores/transcriptions)
- Word confusion pairs and types of errors
- Classifying types of test cases
- Collect real-world noisy data
- Explore training/test contamination issues by attempting to recreate the benchmarks with different audio to test cheating
- Keyterm recall
- Analysis of errors across benchmarks (in the slices tab)
- Multidimensional analysis on the metadata
