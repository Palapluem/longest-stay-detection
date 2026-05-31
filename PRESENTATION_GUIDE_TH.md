# Guideline การนำเสนอ: Longest Stay Detection

เอกสารนี้ใช้เป็นแนวทางสำหรับปรับสไลด์ Canva และอัดคลิปนำเสนอผลงาน coding test

## 1. Audit สไลด์ PDF ปัจจุบัน

ไฟล์ `Longest Stay Detection - Wisit Suwannao.pdf` มี 32 หน้า แต่เนื้อหาส่วนใหญ่ยังเป็นงานเก่าเรื่อง document processing และ topic modeling

สิ่งที่ reuse ได้:

- Slide 1: หน้าปก ปรับชื่อเรียบร้อยแล้ว
- Slide 2: Introduce Myself ใช้ต่อได้
- Slide 31: Q&A ใช้ได้ถ้าต้อง present สด
- Slide 32: Thank You ใช้ปิดคลิปได้
- Visual template: font, spacing และ section-divider style ใช้เป็นแนวต่อได้

สิ่งที่ควรเปลี่ยน:

- Slide 3: Agenda ต้องเปลี่ยนเป็น flow ของ Longest Stay Detection
- Slide 4-30: เนื้อหาเดิมไม่เกี่ยวกับโจทย์นี้ ควรแทนด้วยเนื้อหาใหม่

สำหรับคลิป take-home test ไม่จำเป็นต้องทำ 32 หน้า แนะนำให้เหลือประมาณ 9-11 หน้า เพื่อให้พูดชัดและไม่ยืด

## 2. Target ของคลิป

เวลาที่แนะนำ: 6-8 นาที

สิ่งที่กรรมการควรเข้าใจหลังดูจบ:

1. โจทย์ต้องการหาอะไร
2. ทำไมเลือก YOLO + multi-object tracking
3. นิยาม stationary อย่างไร
4. วัดเวลาอย่างไรโดยไม่ hard-code FPS
5. ผลลัพธ์คือ track ไหนและกี่วินาที
6. มีข้อจำกัดอะไร และจะพัฒนาต่ออย่างไร

## 3. Slide Outline พร้อม Script

### Slide 1: Cover

ชื่อสไลด์:

```text
LONGEST STAY DETECTION
Connected Tech Internship 2026 Take-Home Test
```

เวลาพูด: 10-15 วินาที

Script:

> สวัสดีครับ ผมวิศิษฐ์ สุวรรณเนา วันนี้จะนำเสนอ solution สำหรับโจทย์ Longest Stay Detection โดยเป้าหมายคือหาว่า track ของคนคนไหนหยุดเคลื่อนไหวหรืออยู่ตำแหน่งเดิมนานที่สุดในวิดีโอ และหยุดอยู่นานกี่วินาทีครับ

### Slide 2: Introduce Myself

ใช้ slide profile เดิมได้ แต่พูดสั้น ๆ

เวลาพูด: 15-20 วินาที

Script:

> ผมมีความสนใจด้าน AI และ software engineering โดยงานนี้ผมออกแบบเป็น computer vision pipeline ที่รันได้จริง end-to-end พร้อม annotated video, JSON และ CSV output เพื่อให้ตรวจผลซ้ำได้ครับ

### Slide 3: Problem Statement and Constraints

ใส่ bullet:

- Input: `entrance.mov`
- Detect the person who remained stationary the longest
- Report `track_id`, duration, start time และ end time
- Read FPS and duration from OpenCV metadata
- Output source code, README, requirements, annotated video และ short idea description

เวลาพูด: 35-45 วินาที

Script:

> โจทย์นี้ไม่ได้ถามแค่ว่าในภาพมีคนหรือไม่ แต่ต้องติดตามคนข้าม frame และวัด stationary duration ต่อเนื่อง โดยมี constraint สำคัญคือห้าม hard-code FPS หรือความยาววิดีโอ ผมจึงอ่าน metadata จาก OpenCV และคำนวณ timestamp จาก frame index หารด้วย FPS ครับ

### Slide 4: Definition of Stationary

ใส่หัวใจของแนวคิด:

```text
Stationary = scene position is almost unchanged
```

ใส่สูตร:

```text
foot_point = (center_x, y2)
norm_disp = distance(current_smoothed_foot, old_smoothed_foot) / median_bbox_height
```

เวลาพูด: 45-60 วินาที

Script:

> ผมนิยาม stationary ว่าเป็นการที่ตำแหน่งของคนในฉากแทบไม่เปลี่ยน ไม่ได้หมายความว่าร่างกายต้องนิ่งสนิท เพราะคนอาจขยับแขนหรือศีรษะได้ ผมใช้ foot point จาก bounding box คือ center x กับ y2 เพราะสะท้อนตำแหน่งบนพื้นได้ดีกว่า bbox center จากนั้น normalize ระยะการเคลื่อนที่ด้วย bbox height เพื่อให้ threshold ใช้ได้ดีขึ้นกับทั้งคนใกล้และไกลกล้องครับ

### Slide 5: End-to-End Pipeline

ใช้ภาพ:

```text
assets/longest_stay_pipeline.png
```

เวลาพูด: 70-90 วินาที

Script:

> Pipeline เริ่มจาก input video และอ่าน metadata ด้วย OpenCV จากนั้นใช้ Ultralytics YOLO detect เฉพาะ class person ด้วย classes เท่ากับศูนย์ แล้วใช้ BoT-SORT เป็น tracker หลักเพื่อรักษา track id ข้าม frame แต่สามารถเปลี่ยนเป็น ByteTrack ได้ผ่าน CLI หลังจากนั้นแต่ละ track จะมี history ของ bbox, timestamp และ foot point ผม smooth ตำแหน่งด้วย EMA แล้วใช้ normalized displacement, bbox IoU และ hysteresis เพื่อตัดสิน stationary segment สุดท้ายเลือก track ที่มี stationary segment ยาวที่สุด และ export เป็น annotated video, summary JSON และ tracks CSV ครับ

จุดที่ควรชี้บนภาพ:

- OpenCV Metadata
- `classes=[0]`
- BoT-SORT default / ByteTrack optional
- foot point + EMA
- normalized displacement + IoU + hysteresis
- winner และ output artifacts

### Slide 6: Robustness Decisions

แบ่งเป็น 3 กล่อง:

1. Detection jitter: EMA smoothing + bbox IoU
2. Short interruption: entry/exit hysteresis + missing-gap tolerance
3. Track fragmentation: conservative merge heuristic

เวลาพูด: 50-70 วินาที

Script:

> ในวิดีโอจริง detection box จะมี jitter และบางช่วงอาจหลุด detection สั้น ๆ ผมจึงไม่ตัดสินจาก frame เดียว แต่ใช้ sliding window กับ hysteresis ถ้าสถานะ moving เกิดแค่สั้น ๆ จะยังไม่ปิด segment ทันที ส่วนกรณี tracker แตกเป็นหลาย id ผมเพิ่ม conservative fragment merge โดยดู time gap, ระยะ foot point และความคล้ายของ bbox height เพื่อไม่ merge คนละคนง่ายเกินไปครับ

### Slide 7: Result Preview

ใช้ภาพ:

```text
assets/winner_track_161_at_40s.jpg
```

และเปิด video demo ช่วงประมาณ `36s - 44s` จาก:

```text
results/annotated_entrance.mp4
```

เวลาพูด: 45-60 วินาที

Script:

> นี่คือตัวอย่าง annotated result ที่เวลา 40 วินาทีครับ กรอบสีแดงคือ winner track id 161 ซึ่งอยู่บริเวณด้านขวาของภาพ label แสดงสถานะ stationary และเวลาที่สะสมอยู่ ส่วน track อื่นยังแสดง moving หรือ stationary ตาม state ของแต่ละคน ในคลิปจริงสามารถตรวจย้อนหลังได้ตลอดทั้งวิดีโอครับ

หมายเหตุสำหรับอัดคลิป:

- ไม่ต้องเปิดวิดีโอครบ 85 วินาที
- เปิดเฉพาะช่วงสั้น ๆ ให้เห็น label ขยับและ duration เพิ่มขึ้น
- ซูมหรือขยายฝั่งขวาเล็กน้อยถ้า text บนวิดีโออ่านยาก

### Slide 8: Final Answer

ใช้ตัวเลขใหญ่:

```text
Winner: track_id 161
Longest stationary duration: 41.094 seconds
Interval: 33.699s - 74.793s
```

ใส่ metadata เล็กด้านล่าง:

```text
1920x1080 | 29.883 FPS | 2556 frames | 85.535 seconds
```

เวลาพูด: 30-40 วินาที

Script:

> จากผล run ปัจจุบัน winner คือ track id 161 โดยมี stationary segment ยาวที่สุด 41.094 วินาที ตั้งแต่วินาทีที่ 33.699 ถึง 74.793 ค่า FPS, frame count และ duration ทั้งหมดอ่านจาก OpenCV metadata ไม่มีการ hard-code ครับ

### Slide 9: Deliverables and Repository

ใส่:

- `main.py`
- `0_Longest_Stay_Detection_Report.ipynb`
- `results/annotated_entrance.mp4`
- `results/summary.json`
- `results/tracks.csv`
- GitHub repository link

เวลาพูด: 30-40 วินาที

Script:

> Repository มีทั้ง main.py สำหรับรันผ่าน command line และ notebook walkthrough ภาษาไทยที่แบ่งโค้ดครบทุก step พร้อม executed output นอกจากนี้มี summary JSON, tracks CSV และ annotated video full quality ที่จัดเก็บด้วย Git LFS ครับ

### Slide 10: Limitations and Next Steps

ใส่:

- ID switch during occlusion
- Detector jitter
- Perspective differences
- Camera shake
- Compare `yolo11s/imgsz960` and `bytetrack.yaml`
- Add global motion compensation if needed

เวลาพูด: 40-55 วินาที

Script:

> ข้อจำกัดหลักคือ ID switch ตอนคนบังกัน, detector jitter และ perspective ของกล้อง ซึ่งอาจทำให้ pixel movement ของคนใกล้และไกลต่างกัน แม้จะ normalize ด้วย bbox height แล้วก็ตาม หากพัฒนาต่อ ผมจะเปรียบเทียบ yolo11s ที่ imgsz 960 กับ ByteTrack และเพิ่ม global motion compensation ถ้ากล้องมีการสั่นครับ

### Slide 11: Thank You

เวลาพูด: 10 วินาที

Script:

> ขอบคุณครับ รายละเอียด implementation, วิธีรัน และข้อจำกัดทั้งหมดอยู่ใน GitHub repository ครับ

## 4. Recording Flow ที่แนะนำ

1. เปิดด้วย Canva presentation mode
2. พูด Slide 1-6 ตาม flow โดยยังไม่เปิด code
3. ที่ Slide 7 สลับไปเปิด annotated video ช่วงสั้น ๆ
4. กลับมาสรุป Slide 8-10
5. ปิดด้วย Thank You

ไม่ควรเปิด source code ยาว ๆ ระหว่างคลิป เว้นแต่ requirement ขอให้ walkthrough code โดยตรง เพราะ notebook และ GitHub มีรายละเอียดอยู่แล้ว

## 5. Asset Checklist

- Pipeline diagram: `assets/longest_stay_pipeline.png`
- Winner preview: `assets/winner_track_161_at_40s.jpg`
- Full annotated video: `results/annotated_entrance.mp4`
- JSON summary: `results/summary.json`
- Full notebook walkthrough: `0_Longest_Stay_Detection_Report.ipynb`

## 6. Checklist ก่อนอัดคลิป

- เช็กว่า slide เดิมเรื่อง topic modeling ถูกลบออกหมด
- ใส่ GitHub repository URL ใน slide deliverables
- เปิด video demo ไว้ล่วงหน้าที่ช่วงประมาณ 36 วินาที
- ปิด notification ของเครื่อง
- อัดเสียงทดสอบ 10 วินาทีก่อนเริ่มจริง
- คุมเวลารวมประมาณ 6-8 นาที
- ย้ำว่า `track_id` ไม่ใช่การระบุตัวตนจริงของบุคคล
