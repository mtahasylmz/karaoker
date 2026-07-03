# mutation-stamp: sha256=c459e6f158e77b72764060b758305babefd12493cc6e4e45743695d56ea1af70
# acceptance-mutation-manifest-begin
# {"version":1,"tested_at":"2026-07-02T17:41:32.417307Z","feature_name":"pipeline","feature_path":"features/pipeline.feature","background_hash":"7ac9635d031aee475146f0f701c79c32f2dc6287699f018677a2988e91146658","implementation_hash":"sha256:3580104b0a8c30c909f6e90396082333d22721d3bee1e618cfbe420bcb8c1cfe","scenarios":[{"index":2,"name":"pipeline-3 language hint is honored","scenario_hash":"9f86e56ef8fadb391360da57a5351100e0a8d34a315b5e73318f145fcd75e789","mutation_count":2,"result":{"Total":2,"Killed":2,"Survived":0,"Errors":0},"tested_at":"2026-07-02T17:41:32.417307Z"}]}
# acceptance-mutation-manifest-end

Feature: pipeline
  annemusic turns a music video into karaoke artifacts: an instrumental
  audio track, the isolated vocals stem, line-timed ASS subtitles, and a
  machine-readable manifest. The .ass shows one styled line at a time during
  its sung window — no per-word colour-fill animation. Word-level timings
  still live in manifest.json for machine consumers.

  Background:
    Given the annemusic CLI is installed on a CUDA-capable machine
    And ANNEMUSIC_FIXTURE points at a music video with sung lyrics
    And a matching "lyrics.txt" with the fixture's true lyrics sits beside it
    And ANNEMUSIC_ARTIST and ANNEMUSIC_TITLE name the fixture song

  Scenario: pipeline-1 happy path emits all four artifacts
    When I run "annemusic $ANNEMUSIC_FIXTURE -o out"
    Then the exit code is 0
    And "out/instrumental.wav" is non-silent audio the same duration as the input (±1 s)
    And "out/vocals.wav" exists
    And "out/lyrics.ass" has at least 10 timed Dialogue lines (one per lyric line, no \kf tags)
    And "out/manifest.json" has non-empty "language", "duration", "words", "vocal_activity"

  Scenario: pipeline-2 word timings are sane
    When I run "annemusic $ANNEMUSIC_FIXTURE -o out"
    Then every word in "out/manifest.json" has start < end
    And word starts are non-decreasing
    And every word lies within [0, duration]

  Scenario Outline: pipeline-3 language hint is honored
    When I run "annemusic $ANNEMUSIC_FIXTURE -o out --language <lang>"
    Then the exit code is 0
    And "out/manifest.json" field "language" equals "<lang>"

    Examples:
      | lang |
      | tr   |
      | en   |

  Scenario: pipeline-4 no hint still detects a language
    When I run "annemusic $ANNEMUSIC_FIXTURE -o out"
    Then "out/manifest.json" field "language" is a 2-3 letter code

  Scenario: pipeline-5 known lyrics bias the transcription
    When I run "annemusic $ANNEMUSIC_FIXTURE -o out --lyrics lyrics.txt"
    Then the exit code is 0
    And at least 70% of the distinct words in "lyrics.txt" appear in "out/manifest.json" words

  Scenario: pipeline-6 default output directory is the video stem
    When I run "annemusic $ANNEMUSIC_FIXTURE"
    Then artifacts land in "./<video-stem>/"

  Scenario: pipeline-7 missing input fails fast and clean
    When I run "annemusic does-not-exist.mp4 -o out"
    Then the exit code is non-zero
    And stderr names "does-not-exist.mp4"
    And "out" contains no artifacts

  Scenario: pipeline-8 refusing to clobber
    Given "out/manifest.json" already exists
    When I run "annemusic $ANNEMUSIC_FIXTURE -o out"
    Then the exit code is non-zero and stderr says to pass --force

  Scenario: pipeline-9 language hint is case-insensitive
    When I run "annemusic $ANNEMUSIC_FIXTURE -o out --language TR"
    Then the exit code is 0
    And "out/manifest.json" field "language" equals "tr"

  # Known-lyrics timing: when --artist/--title resolve synced lyrics on
  # LRCLIB, the human line timestamps drive timing (words distributed evenly
  # within each line) instead of ASR + forced alignment. manifest "source"
  # records which path ran: "lrclib" | "asr".

  Scenario: pipeline-10 synced lyrics drive timing
    When I run "annemusic $ANNEMUSIC_FIXTURE -o out --artist $ANNEMUSIC_ARTIST --title $ANNEMUSIC_TITLE"
    Then the exit code is 0
    And "out/manifest.json" field "source" equals "lrclib"
    And "out/manifest.json" has non-empty "words"
    And word starts are non-decreasing and every word lies within [0, duration]
    And "out/lyrics.ass" has at least 10 timed Dialogue lines (one per lyric line, no \kf tags)

  Scenario: pipeline-11 unmatched song falls back to ASR, still succeeds
    When I run "annemusic $ANNEMUSIC_FIXTURE -o out --artist 'No Such Artist 9z9z' --title 'No Such Song 9z9z'"
    Then the exit code is 0
    And "out/manifest.json" field "source" equals "asr"
    And "out/instrumental.wav", "out/vocals.wav", "out/lyrics.ass", "out/manifest.json" all exist

  Scenario: pipeline-12 --no-lyrics-fetch forces the ASR path
    When I run "annemusic $ANNEMUSIC_FIXTURE -o out --artist $ANNEMUSIC_ARTIST --title $ANNEMUSIC_TITLE --no-lyrics-fetch"
    Then the exit code is 0
    And "out/manifest.json" field "source" equals "asr"
