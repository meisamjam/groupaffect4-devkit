$py = "C:\Users\AffectAI\miniconda3\envs\affectai-gpu\python.exe"
$script = "tools/mumt/train_simple.py"
$base = "--split-mode task --test-task T3 --aug-frac 0.3"

Write-Host "=== Fixed sampling-weight aug=0.3, 5 seeds ==="
foreach ($seed in 42,43,44,45,46) {
    $out = & $py $script --split-mode task --test-task T3 --aug-frac 0.3 --seed $seed 2>&1 |
           Select-String "BEST test|Sampling fix|eff_aug_frac"
    Write-Host "seed=$seed done: $out"
}
Write-Host "Done."
