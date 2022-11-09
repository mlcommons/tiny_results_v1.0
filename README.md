# MLPerf™ Tiny v1.0 results

This is the repository containing results and code for the [v1.0 version of the MLPerf™ Tiny benchmark](https://github.com/mlcommons/tiny_results_v1.0).

For benchmark code and rules please see the [GitHub repository](https://github.com/mlcommons/tiny).

Previous versions of the benchmark are available at:
| Version | Results Table | Github Repository |
| ------- | ------------- | ----------------- |
|  v0.7   | [Results](https://mlcommons.org/en/inference-tiny-07/) | [GitHub](https://github.com/mlcommons/tiny_results_v0.7) | 
|  v0.5   | [Results](https://mlcommons.org/en/inference-tiny-05/) | [GitHub](https://github.com/mlcommons/tiny_results_v0.5) |

## MLPerf™ Tiny results directory structure

A submission is for one code base for the benchmarks submitted. An org may make multiple submissions. A submission should take the form of a directory with the following structure. The structure must be followed regardless of the actual location of the actual code, e.g. in the MLPerf repo or an external code host site.

In case of submission of results for multiple systems, please use <system_desc.id> to differentiate these. System names may be arbitrary. We recognize implementatations for multiple systems of the same organization could have different dependencies on a common code base and on each other. When submitting the code, please organize the code as much as possible following a logical structure that makes it possible to reproduce the results, and accompany it with scripts and a README that explains the process. You can use multiple <implementation_id>s to structure your submission.

```
<division>
└── <submitting_organization>
    ├── systems
    │   ├── <system_desc_id>.json #combines hardware and software stack information (one file for each system benchmarked)
    │   ├── TinyMLPerf_v1.0_Submission_Checklist.pdf
    │   └── Energy-Hookup.pdf #image or text description how to reproduce energy configuration and measurment if submitting energy results
    ├── code
    │   └── <benchmark_name per reference>
    │       └── <implementation_id>
    │           └── <Code interface with runner and other arbitrary stuff>
    └── results
        └── <system_desc_id> # (one folder for each system benchmarked)
            └── <benchmark>
                ├── performance
                │   ├── performance_result.txt #results summary produced by runner after performance test
                │   ├── performance_log.txt #log produced by runner after performance test                
                │   └── performance_script.async #script file produced by runner after performance test
                ├── accuracy
                │   ├── accuracy_result.txt #results summmary produced by runner after accuracy test
                │   ├── accuracy_log.txt #log produced by runner after accuracy test
                │   └── accuracy_script.async #script file produced by runner after accuracy test
                └── energy #if submitting energy results
                │   ├── energy_result.txt #results summmary produced by runner after accuracy test
                    ├── energy_log.txt #log produced by runner after accuracy test
                    └── energy_script.async #script file produced by runner after accuracy test
```

System names and implementation names may be arbitrary.

`<division>` must be one of {closed, open}.

`<benchmark>` must be one of {vww, ic, kws, ad}.
