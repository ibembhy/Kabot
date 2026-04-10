$ErrorActionPreference = "Stop"

$root = "C:\Users\cbemb\Documents\Kabot"
$dataDir = Join-Path $root "data"
$btcCsv = Join-Path $dataDir "btcusd_1min.csv"
$python = (Get-Command python).Source

$chunks = @(
    @{
        Label = "part1"
        Start = "2026-03-10T00:00:00Z"
        End   = "2026-03-19T23:59:59Z"
        Csv   = Join-Path $dataDir "kxbtcd_10d_part1.csv"
        Out   = Join-Path $dataDir "kxbtcd_10d_part1.out.log"
        Err   = Join-Path $dataDir "kxbtcd_10d_part1.err.log"
    },
    @{
        Label = "part2"
        Start = "2026-03-20T00:00:00Z"
        End   = "2026-03-29T23:59:59Z"
        Csv   = Join-Path $dataDir "kxbtcd_10d_part2.csv"
        Out   = Join-Path $dataDir "kxbtcd_10d_part2.out.log"
        Err   = Join-Path $dataDir "kxbtcd_10d_part2.err.log"
    },
    @{
        Label = "part3"
        Start = "2026-03-30T00:00:00Z"
        End   = "2026-04-09T23:59:59Z"
        Csv   = Join-Path $dataDir "kxbtcd_10d_part3.csv"
        Out   = Join-Path $dataDir "kxbtcd_10d_part3.out.log"
        Err   = Join-Path $dataDir "kxbtcd_10d_part3.err.log"
    }
)

Set-Location $root

foreach ($chunk in $chunks) {
    Remove-Item $chunk.Csv -ErrorAction SilentlyContinue
    Remove-Item $chunk.Out -ErrorAction SilentlyContinue
    Remove-Item $chunk.Err -ErrorAction SilentlyContinue

    & $python -u -m kabot.cli export-kalshi-history `
        --series KXBTCD `
        --start $chunk.Start `
        --end $chunk.End `
        --btc-csv-path $btcCsv `
        --output-csv $chunk.Csv `
        --progress 1>> $chunk.Out 2>> $chunk.Err
}
