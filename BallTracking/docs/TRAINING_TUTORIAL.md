# BounZ: een eigen basketbal- en ringdetector trainen

Ja: internetvideo's kunnen bruikbaar trainingsmateriaal zijn, maar alleen als je ze mag downloaden en hergebruiken. Gebruik bij voorkeur eigen opnames, Creative Commons-video's, materiaal met expliciete toestemming of een dataset waarvan de licentie training toelaat. Sla steeds de bron en licentie op in een klein tekstbestand naast de video. Download geen video's waarvan de gebruiksvoorwaarden dit verbieden.

Er staan al twee lokaal gedownloade, vrije Pexels-startclips in `data/raw_videos/`: één met spelers/bal/ring en één met een binnenzaal/ringen. Zie [DATA_SOURCES.md](DATA_SOURCES.md) voor de exacte pagina's, makers en downloadmoment. Gebruik deze als aanvulling; je eigen schoolbeelden zijn belangrijker omdat camera, licht en achtergrond dan overeenkomen met je echte test.

Een YOLO-model leert niet rechtstreeks uit een video. De keten is:

```text
legale video -> losse frames -> bounding boxes labelen -> dataset splitsen
-> YOLO trainen -> valideren -> best.pt in de live tracker gebruiken
```

## 1. Kies wat het model moet herkennen

Dit project gebruikt drie klassen. Gebruik deze namen en nummers overal identiek:

| Nummer | Klasse | Wat teken je af? |
| --- | --- | --- |
| `0` | `basketball` | De zichtbare bal, ook als hij deels door een hand of net bedekt is. |
| `1` | `hoop` | De volledige ring; gebruik consequent dezelfde interpretatie. |
| `2` | `white_ping_pong_ball` | Een witte pingpongbal; uitsluitend voor de trackingtest op school. |

Label een pingpongbal dus **niet** als `basketball`. De live tracker toont hem als `WHITE PINGPONG` in cyaan en telt hem niet als basketbalpoging.

## 2. Verzamel gevarieerde video

Gebruik 5–15 korte clips en kies beelden die lijken op je schoolopstelling:

- verschillende afstanden en camerahoeken;
- lichte én donkere achtergrond;
- stilstaande, rollende en vliegende bal;
- beweging, motion blur, gedeeltelijke bedekking door hand/net;
- shots die missen én raken;
- enkele beelden zonder bal of ring (negatieve voorbeelden).

Neem niet iedere frame uit een video: opeenvolgende beelden lijken bijna volledig op elkaar. Ongeveer 500–1.500 gevarieerde, gelabelde beelden is een sterk eerste doel. Houd hele clips apart voor validatie/test; splits nooit willekeurige opeenvolgende frames uit dezelfde clip over train en test, want dat maakt de score te optimistisch.

Plaats je eigen video's lokaal in `data/raw_videos/`. Die map wordt bewust niet naar GitHub gepusht. Gebruik duidelijke bestandsnamen, bijvoorbeeld `school_pingpong_donker_01.mp4` en `school_basketbal_ring_01.mp4`.

### Opnameplan voor de witte pingpongbal op school

Maak minstens zes clips van 20–40 seconden met dezelfde webcam die je later voor BounZ gebruikt:

1. witte bal voor een donkere achtergrond, langzaam rollen;
2. witte bal voor een lichte achtergrond, langzaam rollen;
3. bal gooien met normale snelheid;
4. bal kort bedekt door hand of tafelrand;
5. camera verder weg, zodat de bal klein is;
6. lege scènes zonder bal (negatieve voorbeelden).

Maak daarnaast basketbalclips met de ring zichtbaar. Meng bij het labelen beelden uit de twee Pexels-clips én je eigen clips, maar houd ten minste één volledige eigen clip apart voor de uiteindelijke test.

## 3. Maak frames uit een video

Activeer vanuit de projectroot de virtuele omgeving en voer bijvoorbeeld uit:

```powershell
.\.venv\Scripts\python.exe BallTracking\scripts\extract_frames.py `
  --input BallTracking\data\raw_videos\clip_01.mp4 `
  --output BallTracking\data\frames\clip_01 `
  --every 15 --max-frames 300
```

Bij 30 fps betekent `--every 15` ongeveer twee beelden per seconde. Gebruik voor elke clip een nieuwe uitvoermap. Het script zet de clipnaam voor iedere frame, zodat beelden uit verschillende clips later geen naamconflict hebben. Bekijk de beelden en verwijder slechte of vrijwel identieke beelden voordat je gaat labelen.

## 4. Label elke zichtbare bal en ring

Gebruik bijvoorbeeld CVAT, Roboflow Annotate of Label Studio. Importeer de gekozen JPEG-bestanden en maak de klassen in exact deze volgorde: `basketball`, `hoop`, `white_ping_pong_ball`.

Teken een zo strak mogelijke rechthoek rond elk object. Een labelbestand in YOLO-formaat heeft dezelfde naam als de afbeelding:

```text
frame_00042.jpg
frame_00042.txt
```

Een regel in `frame_00042.txt` ziet er zo uit:

```text
0 0.512500 0.395833 0.062500 0.083333
1 0.486000 0.221000 0.180000 0.090000
2 0.210000 0.660000 0.030000 0.040000
```

Dit betekent `klasse x_middelpunt y_middelpunt breedte hoogte`, waarbij de vier coördinaten tussen 0 en 1 liggen. Exporteer dus als **Ultralytics YOLO / YOLO**. Kopieer de geëxporteerde afbeeldingen naar `data/annotated/images/` en de `.txt`-bestanden naar `data/annotated/labels/`.

Controleer 30 willekeurige beelden zelf. Let vooral op: geen verwisselde klassennummers, geen labels op de schaduw, en geen ontbrekende kleine bal.

## 5. Verdeel de gelabelde dataset

Gebruik 70% trainen, 20% valideren en 10% testen als eerste verdeling:

```powershell
.\.venv\Scripts\python.exe BallTracking\scripts\split_dataset.py `
  --images BallTracking\data\annotated\images `
  --labels BallTracking\data\annotated\labels `
  --output BallTracking\data\dataset
```

De mappen `images/train`, `images/val`, `images/test` en bijhorende `labels/...` worden aangemaakt. Het script gebruikt steeds dezelfde seed en overschrijft nooit bestanden. Voor een eerlijke test reserveer je een hele eigen clip met `--test-prefix`, bijvoorbeeld:

```powershell
.\.venv\Scripts\python.exe BallTracking\scripts\split_dataset.py `
  --images BallTracking\data\annotated\images `
  --labels BallTracking\data\annotated\labels `
  --output BallTracking\data\dataset `
  --test-prefix school_pingpong_test
```

Daarmee belanden alle bestanden die met `school_pingpong_test` beginnen uitsluitend in `test`; de overige beelden worden alleen over train en validatie verdeeld.

## 6. Train lokaal

Controleer eerst of `BallTracking/yolov8n.pt` aanwezig is en of de dataset niet leeg is. Start daarna:

```powershell
.\.venv\Scripts\python.exe BallTracking\training\train.py `
  --epochs 100 --imgsz 960 --batch 8
```

Een NVIDIA-GPU wordt normaal automatisch gebruikt. Zonder GPU duurt training veel langer; gebruik dan bijvoorbeeld `--imgsz 640 --batch 4 --device cpu` als technische proef. Verlaag `--batch` als je een geheugenfout krijgt.

De resultaten komen in `BallTracking/runs/basketball_detector/`. Het belangrijkste bestand is:

```text
BallTracking/runs/basketball_detector/weights/best.pt
```

Vergelijk de resultaten op de **testset** en kijk vooral naar foutpositieven, gemiste kleine ballen en gemiste ringen. Voeg precies die moeilijke situaties daarna toe aan je trainingsdata en train opnieuw.

## 7. Gebruik het getrainde model in de live tracker

Zonder de broncode te veranderen:

```powershell
$env:BOUNZ_MODEL_PATH = "$PWD\BallTracking\runs\basketball_detector\weights\best.pt"
.\.venv\Scripts\python.exe BallTracking\ai\ball_tracking\ball_tracker.py
```

De live tracker verwacht de klassenamen `basketball`, `hoop` en `white_ping_pong_ball`. Ook `basketball hoop`, `rim` en `ping pong ball` worden aanvaard, maar gebruik voor dit project exact de namen uit de YAML. In de live GUI is basketbal oranje, de pingpongbal cyaan, de voorspelling blauw en het spoor rood.

## Kwaliteitscontrole voordat je verder bouwt

- Test op een andere dag, locatie en camera dan de trainingsbeelden.
- Bewaar minstens één volledige video die nooit voor training of validatie is gebruikt.
- Noteer voor elke test: licht, afstand, camera, aantal gemiste ballen en foutpositieven.
- Train niet verder wanneer labels slordig zijn; betere labels zijn waardevoller dan veel bijna identieke frames.
- Controleer de Ultralytics-documentatie voor de precieze YOLO-directorystructuur en trainingsopties: [datasetformaat](https://docs.ultralytics.com/datasets/detect/) en [trainingsmodus](https://docs.ultralytics.com/modes/train/).
