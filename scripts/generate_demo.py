"""Generate the offline demo dataset.

Why a synthetic domain instead of a slice of a public corpus: it needs no
network, it is small enough to train in minutes, and the target behaviour is
checkable (arithmetic, unit conversion, and facts stated in the manual). That
makes it useful as a smoke test for the whole pipeline and as a scale reference
for loss curves.

    python scripts/generate_demo.py --out data/demo

Produces pretrain.jsonl, sft.jsonl, dpo.jsonl and rl.jsonl, all in the formats the
trainers expect.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MANUAL = {
    "Hearth One": {
        "power": "1200 W",
        "capacity": "4.2 L",
        "weight": "5.8 kg",
        "bowl": "stainless steel",
        "warranty": "two years",
        "modes": ["soup", "steam", "brown", "keep warm"],
    },
    "Hearth Mini": {
        "power": "700 W",
        "capacity": "2.0 L",
        "weight": "3.1 kg",
        "bowl": "ceramic coated",
        "warranty": "one year",
        "modes": ["soup", "steam", "keep warm"],
    },
    "Hearth Pro": {
        "power": "1800 W",
        "capacity": "6.5 L",
        "weight": "7.4 kg",
        "bowl": "cast aluminium",
        "warranty": "three years",
        "modes": ["soup", "steam", "brown", "pressure", "keep warm"],
    },
}

FAQ = [
    ("How do I clean the inner bowl?", "Let the bowl cool completely, then wash it with warm water and a soft sponge. Do not use steel wool."),
    ("What does the keep warm mode do?", "It holds the contents between 60 and 75 degrees Celsius for up to twelve hours."),
    ("Why does the lid rattle while cooking?", "The steam valve is releasing pressure. If the rattle is continuous, check that the valve is seated correctly."),
    ("Can I put the bowl in a dishwasher?", "The stainless steel bowl is dishwasher safe. The ceramic coated and cast aluminium bowls must be washed by hand."),
    ("How long does the steam mode run by default?", "The default steam cycle is eighteen minutes. You can add time in one minute steps."),
    ("What should I do if the unit shows error E3?", "E3 means the lid is not closed. Press the lid down until both latches click, then restart the cycle."),
    ("How much water should I add for soup?", "Fill to at least the one litre mark and no higher than two thirds of the bowl."),
    ("Does the appliance need a special outlet?", "It draws less than fifteen amps, so a standard grounded outlet is sufficient."),
    ("What is the delay start range?", "Delay start can be set from thirty minutes to twenty four hours."),
    ("How often should I replace the seal ring?", "Replace the seal ring every eighteen months, or sooner if it no longer seals."),
]

TIPS = [
    "Brown the aromatics on the brown setting before adding liquid. It takes four extra minutes and improves the body of the soup considerably.",
    "Add salt in two stages: a third at the start, the rest after the cycle finishes.",
    "For a thicker soup, remove one cup of solids after cooking, blend it, and return it to the pot.",
    "Keep warm holds food safely for twelve hours, but flavour degrades noticeably after six.",
    "Cut root vegetables into pieces of similar size so they finish at the same time.",
    "Deglaze with a splash of wine or stock after browning to lift the browned bits from the base.",
    "Do not open the lid during the pressure phase. The interlock will hold it shut until pressure drops.",
    "Let the unit rest for ten minutes after a pressure cycle before releasing the remaining steam.",
]


def number(n: int, rng: random.Random) -> str:
    """Sometimes spell numbers out, so the model sees both forms."""
    words = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
             12: "twelve", 18: "eighteen", 24: "twenty four", 30: "thirty"}
    if n in words and rng.random() < 0.3:
        return words[n]
    return str(n)


def arithmetic_items(rng: random.Random, count: int) -> list[tuple[str, str]]:
    items = []
    for _ in range(count):
        kind = rng.choice(["add", "sub", "mul", "div", "mixed"])
        if kind == "add":
            a, b = rng.randint(12, 400), rng.randint(12, 400)
            items.append((f"What is {a} plus {b}?", str(a + b)))
        elif kind == "sub":
            a, b = rng.randint(40, 500), rng.randint(1, 40)
            items.append((f"What is {a} minus {b}?", str(a - b)))
        elif kind == "mul":
            a, b = rng.randint(2, 30), rng.randint(2, 30)
            items.append((f"What is {a} times {b}?", str(a * b)))
        elif kind == "div":
            b = rng.randint(2, 20)
            q = rng.randint(2, 40)
            items.append((f"What is {b * q} divided by {b}?", str(q)))
        else:
            a, b, c = rng.randint(2, 40), rng.randint(2, 40), rng.randint(2, 20)
            items.append((f"What is {a} times {b} plus {c}?", str(a * b + c)))
    return items


def conversion_items(rng: random.Random, count: int) -> list[tuple[str, str]]:
    items = []
    for _ in range(count):
        cups = rng.randint(1, 8)
        items.append((f"How many millilitres are in {cups} cups?", str(round(cups * 236.588, 1))))
        minutes = rng.choice([5, 10, 12, 15, 18, 20, 25, 30])
        items.append((f"How many seconds are in {minutes} minutes?", str(minutes * 60)))
        kilos = rng.choice([1, 2, 3, 5])
        items.append((f"How many grams are in {kilos} kilograms?", str(kilos * 1000)))
    rng.shuffle(items)
    return items[:count]


def pretrain_paragraphs(rng: random.Random) -> list[str]:
    out = []
    for name, spec in MANUAL.items():
        out.append(
            f"The {name} is a countertop cooking appliance. It draws {spec['power']} from a grounded "
            f"outlet and holds {spec['capacity']}. The unit weighs {spec['weight']} and ships with a "
            f"{spec['bowl']} inner bowl. Its cooking modes are {', '.join(spec['modes'])}. "
            f"The warranty covers {spec['warranty']} from the date of purchase."
        )
    for q, a in FAQ:
        out.append(f"Question: {q} Answer: {a}")
    for tip in TIPS:
        out.append(tip)
    for _ in range(600):
        out.append(rng.choice([
            f"Simmer the soup for {number(rng.randint(8, 40), rng)} minutes, then switch to keep warm.",
            f"Brown the onions on the brown setting for {number(rng.randint(3, 9), rng)} minutes before adding stock.",
            f"A {rng.randint(2, 8)} litre batch feeds {number(rng.randint(2, 12), rng)} people.",
            f"Steam vegetables for {number(rng.randint(6, 20), rng)} minutes with the lid latched.",
            f"Add {rng.randint(1, 4)} cups of stock, then reduce by roughly {rng.randint(10, 40)} percent.",
        ]))
    return out


def build_sft(rng: random.Random) -> list[dict]:
    rows = []
    for q, a in FAQ:
        rows.append({"conversations": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})

    for name, spec in MANUAL.items():
        # Repeat each product fact with several phrasings. One example per fact is
        # not enough for a model this size to bind an attribute to a product name:
        # it learns "draws <number> W at full load" and then picks the number at
        # random. Repetition with variation is what makes the binding stick.
        pairs = []
        for _ in range(12):
            pairs += [
                (rng.choice([f"What is the power rating of the {name}?",
                             f"How much power does the {name} draw?",
                             f"Tell me the wattage of the {name}."]),
                 f"The {name} draws {spec['power']} at full load."),
                (rng.choice([f"How much does the {name} hold?",
                             f"What is the capacity of the {name}?"]),
                 f"The {name} has a capacity of {spec['capacity']}."),
                (rng.choice([f"What material is the inner bowl of the {name}?",
                             f"What is the bowl of the {name} made of?"]),
                 f"The {name} ships with a {spec['bowl']} inner bowl."),
                (rng.choice([f"How long is the warranty on the {name}?",
                             f"What warranty covers the {name}?"]),
                 f"The {name} carries a {spec['warranty']} warranty."),
                (rng.choice([f"List the cooking modes of the {name}.",
                             f"Which modes does the {name} have?"]),
                 f"The {name} supports these modes: {', '.join(spec['modes'])}."),
                (rng.choice([f"How much does the {name} weigh?",
                             f"What is the weight of the {name}?"]),
                 f"The {name} weighs {spec['weight']}."),
            ]
        rows.extend({"conversations": [{"role": "user", "content": q},
                                       {"role": "assistant", "content": a}]} for q, a in pairs)

    for q, a in arithmetic_items(rng, 1400):
        rows.append({"conversations": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})
    for q, a in conversion_items(rng, 500):
        rows.append({"conversations": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})

    for _ in range(400):
        name = rng.choice(list(MANUAL))
        # Draw question and answer as a pair. Sampling them independently teaches
        # the model to emit a manual-flavoured sentence unrelated to the question.
        q, a = rng.choice(FAQ)
        rows.append({"conversations": [
            {"role": "system", "content": "You are the support assistant for Hearth countertop cookers. "
                                          "Answer from the manual and keep replies short."},
            {"role": "user", "content": f"I have the {name}. {q}"},
            {"role": "assistant", "content": a},
        ]})

    # Plain FAQ rows without a system prompt, so the model answers manual
    # questions whether or not a system message is present.
    for q, a in FAQ:
        for _ in range(20):
            rows.append({"conversations": [
                {"role": "user", "content": q}, {"role": "assistant", "content": a},
            ]})

    # One tip per row, each tip repeated with a few prompt phrasings so the model
    # does not learn "always answer with the same tip".
    for tip in TIPS:
        for _ in range(12):
            rows.append({"conversations": [
                {"role": "user", "content": rng.choice([
                    "Give me one tip for better soup.",
                    "How can I make the soup taste better?",
                    "Any advice for a better batch?",
                ])},
                {"role": "assistant", "content": tip},
            ]})

    rng.shuffle(rows)
    return rows


def build_dpo(rng: random.Random) -> list[dict]:
    """Chosen answers are on-manual; rejected ones are plausible but wrong."""
    wrong = [
        ("What is the power rating of the Hearth One?", "It uses 2400 W.", "It draws 1200 W at full load."),
        ("What is the capacity of the Hearth Pro?", "It holds 4.2 L.", "The Hearth Pro holds 6.5 L."),
        ("How long is the warranty on the Hearth Mini?", "Three years.", "The Hearth Mini carries a one year warranty."),
        ("What should I do if the unit shows E3?", "Unplug it and call service.", "E3 means the lid is not closed. Latch both sides and restart."),
        ("Can I put the ceramic bowl in the dishwasher?", "Yes, all bowls are dishwasher safe.", "The ceramic coated bowl must be washed by hand."),
        ("How much water for soup?", "Fill the bowl to the top.", "Fill to at least one litre and no more than two thirds of the bowl."),
        ("How long does keep warm hold food?", "Indefinitely.", "Keep warm holds food between 60 and 75 degrees for up to twelve hours."),
        ("What happens if I open the lid during pressure cooking?", "It opens normally.", "The interlock keeps the lid shut until pressure drops."),
    ]
    rows = []
    for q, bad, good in wrong:
        rows.append({
            "chosen": [{"role": "user", "content": q}, {"role": "assistant", "content": good}],
            "rejected": [{"role": "user", "content": q}, {"role": "assistant", "content": bad}],
        })
    for _ in range(200):
        a, b = rng.randint(20, 300), rng.randint(20, 300)
        ans = a + b
        rows.append({
            "chosen": [{"role": "user", "content": f"What is {a} plus {b}?"},
                       {"role": "assistant", "content": str(ans)}],
            "rejected": [{"role": "user", "content": f"What is {a} plus {b}?"},
                         {"role": "assistant", "content": str(ans + rng.choice([-10, -2, 2, 10]))}],
        })
    for _ in range(150):
        q = rng.choice(FAQ)[0]
        rows.append({
            "chosen": [{"role": "user", "content": q}, {"role": "assistant", "content": rng.choice(FAQ)[1]}],
            "rejected": [{"role": "user", "content": q},
                         {"role": "assistant", "content": "I am not sure, please check the box."}],
        })
    rng.shuffle(rows)
    return rows


def build_rl(rng: random.Random) -> list[dict]:
    rows = []
    for q, a in arithmetic_items(rng, 400):
        rows.append({"conversations": [{"role": "user", "content": q}, {"role": "assistant", "content": ""}],
                     "answer": a})
    for q, _ in FAQ[:8]:
        rows.append({"conversations": [{"role": "user", "content": q}, {"role": "assistant", "content": ""}],
                     "answer": ""})
    return rows


def build_mcq(rng: random.Random) -> list[dict]:
    """Multiple-choice rows for `hearthlm.eval mcq`."""
    rows = []
    facts = [
        ("What is the power rating of the Hearth One?", "1200 W", ["1200 W", "700 W", "1800 W", "2400 W"]),
        ("What is the capacity of the Hearth Mini?", "2.0 L", ["2.0 L", "4.2 L", "6.5 L", "1.5 L"]),
        ("Which bowl material ships with the Hearth Pro?", "cast aluminium",
         ["cast aluminium", "stainless steel", "ceramic coated", "glass"]),
        ("How long is the warranty on the Hearth Pro?", "three years",
         ["three years", "one year", "two years", "five years"]),
        ("Which mode is not available on the Hearth Mini?", "pressure",
         ["pressure", "soup", "steam", "keep warm"]),
        ("What does error E3 indicate?", "the lid is not closed",
         ["the lid is not closed", "the bowl is too hot", "low water", "the seal ring is worn"]),
        ("What is the default steam cycle length?", "eighteen minutes",
         ["eighteen minutes", "twelve minutes", "thirty minutes", "nine minutes"]),
        ("What temperature range does keep warm hold?",
         "60 to 75 degrees", ["60 to 75 degrees", "40 to 50 degrees", "80 to 95 degrees", "90 to 100 degrees"]),
    ]
    for q, answer, options in facts:
        opts = options[:]
        rng.shuffle(opts)
        rows.append({"question": q, "options": opts, "answer": "ABCD"[opts.index(answer)]})
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/demo")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def dump(name: str, rows: list[dict]) -> None:
        path = out / name
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(rows)} rows")

    passages = pretrain_paragraphs(rng)
    sft_rows = build_sft(rng)

    dump("pretrain.jsonl", [{"text": t} for t in passages])
    dump("sft.jsonl", sft_rows)
    dump("dpo.jsonl", build_dpo(rng))
    dump("rl.jsonl", build_rl(rng))
    dump("mcq.jsonl", build_mcq(rng))

    # Pretraining corpus that includes the conversations in rendered form. A
    # model this small sees the chat markers for the first time during SFT if you
    # skip this, and spends its whole SFT budget learning the template instead of
    # the answers. Documented as step 3 of the README walkthrough.
    from hearthlm.data.chat_template import render_text

    mixed = [{"text": t} for t in passages]
    for row in sft_rows:
        mixed.append({"text": render_text(row["conversations"], enable_thinking=False)})
    dump("pretrain_mix.jsonl", mixed)


if __name__ == "__main__":
    main()