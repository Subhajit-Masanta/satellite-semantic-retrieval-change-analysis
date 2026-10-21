# Reads the real page number of every heading from Word and writes pages.json,
# which make_report.py then uses to fill the Table of Contents.
#   powershell -File report_tools\get_pages.ps1
$root = Split-Path -Parent $PSScriptRoot
$f = Join-Path $root "BCSE407L_Project_Report_Satellite_Semantic_Retrieval_v3_draft.docx"
$w = New-Object -ComObject Word.Application; $w.Visible = $false
$doc = $w.Documents.Open($f, $false, $true)
$doc.Repaginate()
$map = [ordered]@{}
foreach ($p in $doc.Paragraphs) {
  $t = $p.Range.Text.Trim()
  if ($t.Length -gt 3 -and $t.Length -lt 70 -and $t -match '^\d+\.(\d+(\.\d+)?)? ?\S') {
    if ($p.Range.Information(12)) { continue }          # skip the TOC table itself
    $key = $t -replace '^\d+\. ', ''                    # "1. INTRODUCTION" -> "INTRODUCTION"
    if (-not $map.Contains($key)) { $map[$key] = $p.Range.Information(1) }
  }
}
$total = $doc.ComputeStatistics(2)
$doc.Close($false); $w.Quit()
$json = $map | ConvertTo-Json
[System.IO.File]::WriteAllText((Join-Path $PSScriptRoot "pages.json"), $json, (New-Object System.Text.UTF8Encoding($false)))
"total pages: $total"
"DATASET $($map['DATASET DETAILS'])  IMPLEMENTATION $($map['IMPLEMENTATION'])  RESULTS $($map['RESULTS AND ANALYSIS'])  REFERENCES $($map['REFERENCES'])"
