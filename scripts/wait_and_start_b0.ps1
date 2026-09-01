# Waiter: block until all mesonet-related python processes exit, then start the efficientnet_b0 queue.
# Keeps max 2 concurrent trainings on this 16GB-RAM laptop to avoid the OOM native-crash loop
# observed when 3 queues ran in parallel (see results/baseline_honest/protocol.md).
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'mesonet' }) {
    Start-Sleep -Seconds 60
}
Set-Location 'C:\Users\86188\Desktop\trae'
& 'D:\python\python.exe' scripts/run_honest_queue.py --models efficientnet_b0 2>&1 |
    Out-File -FilePath 'results\baseline_honest\b0_queue.log' -Append -Encoding utf8
