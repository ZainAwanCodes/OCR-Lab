import re

def normalize_text(text: str, ignore_case: bool = True, ignore_punctuation: bool = False) -> str:
    """
    Standardizes whitespace, case, and optional punctuation for fair evaluation.
    """
    if text is None:
        return ""
    res = text.strip()
    if ignore_case:
        res = res.lower()
    if ignore_punctuation:
        res = re.sub(r"[^\w\s]", "", res)
    # Collapse multiple whitespaces/newlines into single spaces
    res = re.sub(r"\s+", " ", res)
    return res

def levenshtein_matrix(reference_tokens: list, hypothesis_tokens: list) -> tuple[int, int, int, int]:
    """
    Computes dynamic programming matrix for Levenshtein distance.
    Returns (distance, substitutions, deletions, insertions).
    """
    n = len(reference_tokens)
    m = len(hypothesis_tokens)

    # dp[i][j] stores (cost, S, D, I)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    # Initializing base cases
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)  # Deletions from reference
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)  # Insertions into hypothesis

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ref_tok = reference_tokens[i - 1]
            hyp_tok = hypothesis_tokens[j - 1]

            if ref_tok == hyp_tok:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Substitution
                cost_sub, s, d, ins = dp[i - 1][j - 1]
                cand_sub = (cost_sub + 1, s + 1, d, ins)

                # Deletion
                cost_del, s_d, d_d, ins_d = dp[i - 1][j]
                cand_del = (cost_del + 1, s_d, d_d + 1, ins_d)

                # Insertion
                cost_ins, s_i, d_i, ins_i = dp[i][j - 1]
                cand_ins = (cost_ins + 1, s_i, d_i, ins_i + 1)

                dp[i][j] = min(cand_sub, cand_del, cand_ins, key=lambda x: x[0])

    return dp[n][m]

def calculate_error_rate(reference: str, hypothesis: str, mode: str = "word", ignore_case: bool = True) -> dict:
    """
    Calculates WER or CER based on Levenshtein edit distance.
    mode: 'word' or 'char'
    Formula: Error Rate = (Substitutions + Deletions + Insertions) / N_reference
    """
    ref_norm = normalize_text(reference, ignore_case=ignore_case)
    hyp_norm = normalize_text(hypothesis, ignore_case=ignore_case)

    if mode == "word":
        ref_tokens = ref_norm.split() if ref_norm else []
        hyp_tokens = hyp_norm.split() if hyp_norm else []
    else:
        # Character mode: remove space characters or compare characters directly
        ref_tokens = list(ref_norm.replace(" ", ""))
        hyp_tokens = list(hyp_norm.replace(" ", ""))

    num_ref = len(ref_tokens)
    num_hyp = len(hyp_tokens)

    if num_ref == 0:
        if num_hyp == 0:
            return {
                "rate": 0.0,
                "percentage_str": "0.0%",
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "distance": 0,
                "reference_count": 0,
                "hypothesis_count": 0
            }
        else:
            return {
                "rate": 1.0,
                "percentage_str": "100.0% (Empty reference)",
                "substitutions": 0,
                "deletions": 0,
                "insertions": num_hyp,
                "distance": num_hyp,
                "reference_count": 0,
                "hypothesis_count": num_hyp
            }

    dist, subs, dels, inss = levenshtein_matrix(ref_tokens, hyp_tokens)
    error_rate = dist / num_ref
    pct = round(error_rate * 100, 2)

    return {
        "rate": round(error_rate, 4),
        "percentage": pct,
        "percentage_str": f"{pct:.2f}%",
        "substitutions": subs,
        "deletions": dels,
        "insertions": inss,
        "distance": dist,
        "reference_count": num_ref,
        "hypothesis_count": num_hyp
    }

def compare_results(tesseract_text: str, ai_text: str, ground_truth: str = None, ignore_case: bool = True) -> dict:
    """
    Produces a comprehensive, mathematically sound comparative report.
    """
    t_clean = (tesseract_text or "").strip()
    a_clean = (ai_text or "").strip()
    gt_clean = (ground_truth or "").strip() if ground_truth else None

    # Text match between Tesseract and AI
    exact_match_t_vs_a = (
        normalize_text(t_clean, ignore_case=ignore_case) == 
        normalize_text(a_clean, ignore_case=ignore_case)
    )

    t_words = len(t_clean.split()) if t_clean else 0
    t_chars = len(t_clean)
    a_words = len(a_clean.split()) if a_clean else 0
    a_chars = len(a_clean)

    report = {
        "tesseract_metrics": {
            "char_count": t_chars,
            "word_count": t_words
        },
        "ai_metrics": {
            "char_count": a_chars,
            "word_count": a_words
        },
        "exact_match_between_ocr": exact_match_t_vs_a,
        "has_ground_truth": bool(gt_clean),
        "notes": [
            "Character count and word count quantify output volume, not linguistic accuracy.",
            "A higher character count does not indicate higher accuracy without ground-truth reference."
        ]
    }

    if gt_clean:
        # Calculate metrics against ground truth for Tesseract
        t_wer = calculate_error_rate(gt_clean, t_clean, mode="word", ignore_case=ignore_case)
        t_cer = calculate_error_rate(gt_clean, t_clean, mode="char", ignore_case=ignore_case)

        # Calculate metrics against ground truth for AI
        a_wer = calculate_error_rate(gt_clean, a_clean, mode="word", ignore_case=ignore_case)
        a_cer = calculate_error_rate(gt_clean, a_clean, mode="char", ignore_case=ignore_case)

        report["evaluation"] = {
            "ground_truth_stats": {
                "word_count": len(normalize_text(gt_clean, ignore_case=ignore_case).split()),
                "char_count": len(normalize_text(gt_clean, ignore_case=ignore_case).replace(" ", ""))
            },
            "tesseract": {
                "wer": t_wer,
                "cer": t_cer,
                "exact_match_gt": normalize_text(t_clean, ignore_case=ignore_case) == normalize_text(gt_clean, ignore_case=ignore_case)
            },
            "ai": {
                "wer": a_wer,
                "cer": a_cer,
                "exact_match_gt": normalize_text(a_clean, ignore_case=ignore_case) == normalize_text(gt_clean, ignore_case=ignore_case)
            }
        }
    else:
        report["evaluation"] = None
        report["ground_truth_message"] = (
            "No ground-truth reference text provided. To reliably calculate Word Error Rate (WER) "
            "and Character Error Rate (CER), paste the verified reference transcription above."
        )

    return report
