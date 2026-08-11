# Annotator guide

You have one file: `annotate_a1.html`, `annotate_a2.html` or `annotate_a3.html`. Open it in a
browser. It is the whole task — there is no login, no website and no dashboard.

You need no biomedical background. Everything you need is on the screen.

> **The one rule that matters most:** judge what the highlighted text **says**, not what is true.
> You are not checking medicine. You are checking whether the quoted passage backs up the sentence.

---

## 1. What you are doing, and why it is built this way

A system answers a biomedical question using retrieved passages, and it splits its answer into
short factual sentences — **claims** — each pointing at the passage text it came from. Your job is
to say, claim by claim, whether the pointed-at text actually supports it.

Three people label the **same** set, in the **same** order, independently. That is deliberate: the
measurement we need is how much three careful readers *agree*, and agreement only means something
if nobody influenced anybody.

So, two requests:

- **Do not discuss specific claims or labels with the other two annotators** while you are working.
  Questions about the *rules* are welcome and useful — ask the maintainer, not each other.
- **Work in the order the form gives you.** Do not skip ahead. The form enforces this. If you stop
  halfway, an unbroken run from the start is still usable; a scattered half is not.

Your file does not tell you which system produced a claim, and it should not. Do not try to work it
out.

---

## 2. The four labels

For every highlighted passage span, pick exactly one:

| Label | Meaning |
|---|---|
| `SUPPORTED` | The span asserts the claim. |
| `PARTIAL` | The span asserts **part** of the claim, or asserts it more weakly or more narrowly than the claim states. |
| `NOT_SUPPORTED` | The span does not address the claim. |
| `CONTRADICTED` | The span asserts the **opposite** of the claim. |

### Use no outside knowledge

A claim can be **false but `SUPPORTED`** — the passage says it, so that is what you record. A claim
can be **true but `NOT_SUPPORTED`** — correct medicine, absent from the span. If you happen to know
the literature, set that aside. Anything you had to know from outside the span is not support.

### `SUPPORTED` vs `PARTIAL` is the hard boundary

`NOT_SUPPORTED` vs `CONTRADICTED` is nearly mechanical: does the span assert the opposite, or say
nothing on the point? The judgement that costs real attention is the first one. Use `PARTIAL`
whenever the span does not carry the **whole** claim:

| Claim | Span says | Label | Why |
|---|---|---|---|
| Metformin reduced HbA1c by 1.2%. | "Metformin reduced HbA1c by 1.2% at 12 weeks." | `SUPPORTED` | Whole claim, and more. |
| Metformin reduced HbA1c by 1.2%. | "Metformin reduced HbA1c." | `PARTIAL` | The number is not there. |
| Metformin reduced HbA1c in all adults. | "Metformin reduced HbA1c in adults over 65." | `PARTIAL` | Narrower population than claimed. |
| Metformin reduced HbA1c by 1.2%. | "Metformin reduced HbA1c by 0.4%." | `PARTIAL`, not `CONTRADICTED` | A smaller effect is a weaker version, not the opposite. |
| Metformin reduced HbA1c. | "Metformin did not reduce HbA1c." | `CONTRADICTED` | The opposite. |
| Metformin reduced HbA1c. | "No effect on weight was observed." | `NOT_SUPPORTED` | Different outcome; the claim is not addressed. |

Three recurring traps, all of which land on `PARTIAL` or `CONTRADICTED` rather than `SUPPORTED`:

- **Numbers.** A different dose, duration, percentage or sample size is a different assertion.
- **Population and scope.** "in adults", "in all patients", "in the elderly" are part of the claim.
- **Strength of language.** "may reduce" does not assert "reduces"; "was associated with" does not
  assert "causes".

If you genuinely cannot decide between two labels, pick the weaker one and **write why in the notes
box**. The notes are read; the disagreements we can explain are worth more than a tidy guess.

---

## 3. Two more judgements per claim

### Is the claim well-formed? (`claim_validity`)

A separate, binary question: **could this sentence be checked at all, on its own?**

Not about truth, and not about support. It asks only whether the machine produced something
judgeable. Mark it **not well-formed** when the claim is a fragment, a question, two assertions
crammed together, or still depends on something outside itself:

| Claim | Well-formed? | Why |
|---|---|---|
| "Metformin reduces all-cause mortality in patients with type 2 diabetes." | Yes | Self-contained. |
| "It reduces all-cause mortality." | **No** | "It" is unresolved. |
| "This was not observed in the elderly." | **No** | "This" is unresolved. |
| "Metformin reduces mortality and improves glycaemic control." | **No** | Two claims in one. |
| "in patients over 75" | **No** | A fragment. |

**Still answer the support questions either way.** A malformed claim gets both: not well-formed,
and your best reading of the support.

**One special case: a claim that declines to answer is well-formed.** Something like *"The question
of whether prophylaxis helps all patients is not addressed by the provided passages."* is a correct
thing for a system to say. Mark it **well-formed**. Note the difference: *"Metformin does not reduce
mortality"* is an assertion about the world and is labelled for support normally; *"the passages do
not mention mortality"* is a statement about the passages themselves.

### All spans together (the union judgement)

Some claims cite two or three spans. After labelling each span on its own, read **all of that
claim's spans together as one piece of evidence** and label the claim against the whole.

This is a real, separate judgement, not a summary of the others. Two citations can be **jointly
necessary** — a dose from one span and an outcome from another. Each alone is `PARTIAL`; together
they can be `SUPPORTED`. Record that.

---

## 4. Using the form

1. **Open your file** in a browser. Use **one browser on one machine** for the whole task.
2. Work top to bottom. Answer every question on every claim in the current question.
3. Press **Mark question complete →**. It stays disabled until nothing is unanswered.
4. **Press "Download my labels (JSONL)" at the end of every session** and send that file in. It is
   your own copy as well as our data.

The header tells you where you stand:

```
Annotator a1   Question 7 of 62 · 6 complete        backed up 14:22:31   saved 14:22:28
```

- `saved` — written into this browser. This happens on every click.
- `backed up` — also copied to the lab machine on the LAN. If it says **no backup — working
  offline**, keep going: nothing is lost, and it will catch up when the machine is reachable.

The form measures how long you actually spend on each question. That is for costing the task
honestly, never for judging your pace. Take breaks; the clock stops when you leave a question.

### If something goes wrong

Press **Restore…**. You get every copy we can find, each labelled with how many questions it holds
and when it was saved:

- **This browser's copy**
- **The copy on the collector** — what the lab machine has
- **A file you exported** — any JSONL you downloaded earlier

Pick one. **The copy you pick replaces the others — nothing is merged for you.** Read the counts
before choosing. If two copies look close, take the one with more completed questions.

This is also how you move to a different machine: open your form there and restore.

### Please do not

- Do not open your form in two browsers or two machines at once. Two half-passes cannot be joined.
- Do not edit the HTML file, or rename it.
- Do not send your file to the other annotators.
- If you are ever handed a `keyfile.jsonl`, you were sent it by mistake. Do not open it, and say so.

---

## 5. Time, and the schedule

The full set is roughly **250 claims across ~62 questions**, and we cost it at **10–16 hours** per
annotator across the annotation window. It is unpaid, careful reading, and that number is our
estimate, not a measurement — which is exactly what the pilot exists to check.

- **Pilot** — a short first run on a handful of questions. Its purpose is to find the places where
  these rules are unclear, and to see whether the hours are real. Say what confused you; that
  feedback changes the guide before the main pass.
- **Main pass** — the full set, in the order your form gives you.

Download and send your JSONL at the end of every session, not only at the end of the task. It is
how we know the pass is progressing, and it is your backup.

---

## 6. Where to ask

Questions about **the rules** — what counts as `PARTIAL`, whether a claim is well-formed, how to
treat a span — go to the maintainer, and the answer goes to all three of you at once so you are
working from the same rules. Questions about **specific claims in your file** also go to the
maintainer, never to the other annotators.

If the answer changes how a case should be labelled, it will be added to this guide, and you will be
told which earlier questions to revisit.
