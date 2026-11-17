# GTA Donat

Десктопний прототип для Windows, який накладає нікнейми й суми донатів над персонажами на записі геймплею. YOLO11n виконує інференс на NVIDIA GPU, а TrackTrack зіставляє NPC між кадрами.

## Встановлення

У PowerShell у корені репозиторію запусти команди:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Запуск

Додай `gameplay.mp4` у корінь репозиторію. Під час першого запуску Ultralytics завантажить `yolo11n.pt`.

```powershell
.\.venv\Scripts\python.exe main.py
```

Для відео з іншою назвою можна явно вказати вхідний і вихідний файли:

```powershell
.\.venv\Scripts\python.exe main.py --input 1117.mp4 --output output_preview.mp4
```

Лівий клік по персонажу перемикає видимість його бейджа. `Space` ставить прев’ю на паузу або продовжує відтворення, а `Q` завершує запис. Прев’ю за замовчуванням зберігається в `output_preview.mp4` як відеодоріжка з бейджами.
