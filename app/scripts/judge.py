import os
import json
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.scripts.config import (
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY,
    EVAL_MODEL,
    REPORTS_DIR_BASE,
    JUDGE_PROMPT_PATH,
)
from app.scripts.evaluator import load_ground_truths


def _read_report_lines(file_path: str) -> List[Dict[str, Any]]:
    """Helper to read jsonl records safely without circular imports."""
    records = []
    if not os.path.exists(file_path):
        return records
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def _load_judge_prompt() -> str:
    """Loads the judge system prompt from system_prompt/judge_prompt.txt."""
    with open(JUDGE_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


JUDGE_SYSTEM_PROMPT = _load_judge_prompt()


def get_judge_score_file(report_file: str) -> str:
    """Returns the path of the judge evaluation score file (.jsonl) for a given report."""
    dirname = os.path.dirname(report_file)
    basename = os.path.basename(report_file).replace("report_", "judge_")
    return os.path.join(dirname, basename)


def load_judge_summary(report_file: str) -> Optional[Dict[str, Any]]:
    """
    Loads judge scores (.jsonl) and computes an aggregated summary.
    Returns None if the judge score file does not exist or has no scores.
    """
    judge_file = get_judge_score_file(report_file)
    if not os.path.exists(judge_file):
        return None
    try:
        with open(judge_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
    except Exception:
        return None
    if not lines:
        return None
    total = len(lines)
    avg_emp = round(sum(s.get("empathy", 0) for s in lines) / total, 2)
    avg_act = round(sum(s.get("actionability", 0) for s in lines) / total, 2)
    avg_con = round(sum(s.get("conciseness", 0) for s in lines) / total, 2)
    avg_qual = round(sum(s.get("quality_pct", 0) for s in lines) / total, 1)
    return {
        "report_file": os.path.basename(report_file),
        "eval_model": EVAL_MODEL,
        "total_judged": total,
        "avg_quality_pct": avg_qual,
        "avg_empathy": avg_emp,
        "avg_actionability": avg_act,
        "avg_conciseness": avg_con,
    }


def score_single_response(
    client: OpenAI,
    eval_model: str,
    question_id: Any,
    customer_message: str,
    suggested_response: str,
) -> Optional[Dict[str, Any]]:
    """Asks the judge model to score a single suggested response on a 1-5 scale."""
    user_prompt = f"""[Customer Message]:
{customer_message}

[Suggested Support Response]:
{suggested_response}

Evaluate the suggested response and output the JSON score."""

    try:
        response = client.chat.completions.create(
            model=eval_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()

        # Ensure we have some content
        if not content:
            raise ValueError("Empty response from judge model")

        # Strip possible markdown code fences
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        # Try to parse JSON; fallback to extracting JSON via regex
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*?\}", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("Judge output did not contain valid JSON")

        empathy = int(data.get("empathy", 3))
        actionability = int(data.get("actionability", 3))
        conciseness = int(data.get("conciseness", 3))

        total_pts = empathy + actionability + conciseness
        pct = round((total_pts / 15.0) * 100, 1)

        return {
            "question_id": question_id,
            "empathy": empathy,
            "actionability": actionability,
            "conciseness": conciseness,
            "quality_pct": pct,
            "critique": data.get("critique", ""),
        }
    except Exception as e:
        # Return None to indicate a failure; caller will skip this entry
        return None


def evaluate_report_quality(
    report_file: str,
    client: Optional[OpenAI] = None,
    eval_model: str = EVAL_MODEL,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Evaluates all responses in a report using the local judge model.
    Saves scores incrementally to .jsonl and skips already judged question_ids.
    Gracefully returns None if judge model is not yet pulled/available.
    """
    judge_score_file = get_judge_score_file(report_file)

    if force and os.path.exists(judge_score_file):
        os.remove(judge_score_file)

    records = _read_report_lines(report_file)
    if not records:
        return None

    # Load already-judged question IDs for restart/resume mechanism
    judged_ids = set()
    if os.path.exists(judge_score_file):
        try:
            with open(judge_score_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        if "question_id" in record:
                            judged_ids.add(record["question_id"])
        except Exception:
            pass

    # If all test cases with suggested responses are already judged, return cached summary directly
    cases_to_judge = [
        r for r in records
        if (r.get("output") or {}).get("suggested_response") and r.get("question_id") not in judged_ids
    ]

    if not cases_to_judge and judged_ids:
        print(f"All {len(judged_ids)} responses already judged for: {os.path.basename(report_file)}")
        return load_judge_summary(report_file)

    if client is None:
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

    # Pre-check if judge model is available on Ollama by testing a ping/call
    try:
        test_ping = client.chat.completions.create(
            model=eval_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
    except Exception as e:
        print(f"Notice: Judge model '{eval_model}' is not available yet. (Error: {e})")
        print(f"To enable quality evaluation, run: ollama run {eval_model}")
        return None

    ground_truths = load_ground_truths()
    print(
        f"Judging {len(cases_to_judge)} remaining responses (already judged: {len(judged_ids)}) "
        f"with '{eval_model}' for: {os.path.basename(report_file)}..."
    )

    with open(judge_score_file, "a", encoding="utf-8") as out_f:
        for idx, rec in enumerate(cases_to_judge, start=1):
            output = rec.get("output") or {}
            suggested = output.get("suggested_response", "")
            q_id = rec.get("question_id")
            cust_msg = ground_truths.get(q_id, {}).get("message", "")

            print(f"Checking single score for question ID : {q_id}")
            score_res = score_single_response(client, eval_model, q_id, cust_msg, suggested)
            print(f"Score:{score_res}")

            # Immediately append valid score to file line-by-line
            if score_res is not None:
                out_f.write(json.dumps(score_res) + "\n")
                out_f.flush()
                judged_ids.add(q_id)

            if idx % 10 == 0 or idx == len(cases_to_judge):
                print(f"  Processed {idx}/{len(cases_to_judge)} remaining...")

    return load_judge_summary(report_file)


def run_judge_on_all_reports(force: bool = False):
    """Discovers all report files and evaluates quality with the judge model."""
    import glob
    pattern = os.path.join(REPORTS_DIR_BASE, "*", "report_temp_*.jsonl")
    report_files = sorted(glob.glob(pattern))

    if not report_files:
        print("No report files found to evaluate.")
        return

    print("=" * 60)
    print(f"RUNNING LLM-AS-A-JUDGE QUALITY EVALUATION ({EVAL_MODEL})")
    print("=" * 60)

    for report_file in report_files:
        res = evaluate_report_quality(report_file, force=force)
        if res:
            print(f"Done: {report_file} -> Quality Score: {res['avg_quality_pct']}%")
        print("-" * 60)


if __name__ == "__main__":
    import sys
    force_run = "--force" in sys.argv
    run_judge_on_all_reports(force=force_run)
