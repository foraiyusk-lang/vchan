# Start training as a fully detached process (survives the Claude session / shell closing)
$root = "C:\Users\ysk72\reimu_ai_vc"
Start-Process -FilePath "$root\.venv\Scripts\python.exe" `
  -ArgumentList "scripts\run_training.py","reimu_v1","data\train_v1","--stages","train","--epochs","40","--save-every","2","--batch","1" `
  -WorkingDirectory $root -WindowStyle Hidden `
  -RedirectStandardOutput "$root\logs\train_detached.out" -RedirectStandardError "$root\logs\train_detached.err"
