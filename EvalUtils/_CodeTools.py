from evalplus.evaluate import *

def evaluate_minimal(
    dataset: str,
    samples: dict,
    base_only: bool = False,
    parallel: Optional[int] = None,
    test_details: bool = False,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
    mini: bool = False,
    noextreme: bool = False,
    version: str = "default"
):
    n_workers = parallel or max(1, multiprocessing.cpu_count() // 2)

    if dataset == "humaneval":
        problems = get_human_eval_plus(
            mini=mini, noextreme=noextreme, version=version
        )
        dataset_hash = get_human_eval_plus_hash(
            mini=mini, noextreme=noextreme, version=version
        )
        expected_output = get_groundtruth(problems, dataset_hash, [])
    elif dataset == "mbpp":
        problems = get_mbpp_plus(mini=mini, noextreme=noextreme, version=version)
        dataset_hash = get_mbpp_plus_hash(
            mini=mini, noextreme=noextreme, version=version
        )
        expected_output = get_groundtruth(
            problems,
            dataset_hash,
            MBPP_OUTPUT_NOT_NONE_TASKS,
        )

    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "hash": dataset_hash,
        "eval": {},
    }

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        eval_results = defaultdict(list)  # task_id ->
        remainings = set()

        print("Reading samples...")
        for sample in tqdm(samples):
            task_id = sample["task_id"]
            if task_id not in problems:
                warn(
                    f"Task {task_id} is found in the samples but not found in the dataset"
                )
                continue
            solution = (
                sample["solution"]
                if "solution" in sample
                else problems[task_id]["prompt"] + sample["completion"]
            )
            remainings.add(sample["task_id"])
            args = (
                dataset,
                completion_id[task_id],
                problems[task_id],
                solution,
                expected_output[task_id],
                base_only,
                not test_details,  # fast_check
                sample["task_id"],
                min_time_limit,
                gt_time_limit_factor,
            )
            futures.append(executor.submit(check_correctness, *args))
            completion_id[task_id] += 1
            n_samples += 1

        assert n_samples == len(remainings), "Missing problems in unfinished"
        assert len(completion_id) == len(problems), "Missing problems in samples"

        def stucking_checker():
            while remainings:
                last_size = len(remainings)
                time.sleep(20)
                if last_size != len(remainings) or len(remainings) == 0:
                    continue
                # Potential stucking
                warn("No samples had finished testing in the last 20s")
                warn(f"{len(remainings)} samples to be tested: {remainings}")

        threading.Thread(target=stucking_checker).start()

        for future in tqdm(as_completed(futures), total=n_samples):
            result = future.result()
            remainings.remove(result["_identifier"])
            eval_results[result["task_id"]].append(result)

    # sort the results for each problem by completion_id
    for task_id, task_results in eval_results.items():
        task_results.sort(key=lambda x: x["completion_id"])
        results["eval"][task_id] = []
        for res in task_results:

            def get_failed_tests(stat, details, inputs) -> List[Any]:
                if stat == PASS or not details:
                    return []

                if test_details:
                    return [
                        inputs[i] for i in range(len(details)) if not details[i]
                    ]

                # else => simply return the only and the last fail test
                return [inputs[len(details) - 1]]

            base_stat, base_details = res["base"]
            base_fail_tests = get_failed_tests(
                base_stat, base_details, problems[task_id]["base_input"]
            )

            # initialize plus tests
            plus_stat = None
            plus_fail_tests = []

            # with plus tests
            if not base_only:
                plus_stat, plus_details = res["plus"]
                plus_fail_tests = get_failed_tests(
                    plus_stat, plus_details, problems[task_id]["plus_input"]
                )

            if dataset == "mbpp":
                base_fail_tests = mbpp_serialize_inputs(task_id, base_fail_tests)
                plus_fail_tests = mbpp_serialize_inputs(task_id, plus_fail_tests)

            results["eval"][task_id].append(
                {
                    "task_id": task_id,
                    "solution": res["solution"],
                    "base_status": base_stat,
                    "plus_status": plus_stat,
                    "base_fail_tests": base_fail_tests,
                    "plus_fail_tests": plus_fail_tests,
                }
            )

    # Calculate pass@k.
    total = np.array([len(r) for r in results["eval"].values()])
    base_correct = []
    new_correct = []

    for res in results["eval"].values():
        bc = sum([r["base_status"] == PASS for r in res])
        base_correct.append(bc)
        if not base_only:
            new_correct.append(
                sum(
                    [
                        res[i]["base_status"] == res[i]["plus_status"] == PASS
                        for i in range(len(res))
                    ]
                )
            )
    base_correct = np.array(base_correct)

    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, base_correct, k).mean()
        for k in [1, 10, 100]
        if total.min() >= k
    }
    # cprint(f"{dataset} (base tests)", "red")
    # for k, v in pass_at_k.items():
    #     cprint(f"{k}:\t{v:.3f}", "red")
        
    pass_at_k_base = pass_at_k
    
    if new_correct:
        # cprint(f"{dataset}+ (base + extra tests)", "green")
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, np.array(new_correct), k).mean()
            for k in [1, 10, 100]
            if (total >= k).all()
        }
        # for k, v in pass_at_k.items():
        #     cprint(f"{k}:\t{v:.3f}", "green")
        
        return {"base": pass_at_k_base, "+": pass_at_k}
    return {"base": pass_at_k_base}
    