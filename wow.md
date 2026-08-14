 ## 1. Windows에 Python 설치

  1. Python 공식 설치파일 (https://www.python.org/downloads/windows/) 설치
  2. 설치 화면에서 Add Python to PATH 체크
  3. Git을 사용할 경우 Git for Windows (https://git-scm.com/download/win) 설치

  PowerShell:

  git clone https://github.com/<GitHub계정명>/imutest.git
  cd imutest

  py -3 -m venv .venv
  .\.venv\Scripts\Activate.ps1

  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt

  PowerShell에서 실행 권한 오류가 나면:

  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

  실행:

  python map.py
  python cv.py

  영상 분석은 다음처럼 실행한 뒤 GUI에서 Browse로 선택하면 됩니다.

  python cv.py

  Windows용 Python에는 보통 tkinter가 포함되므로 Linux의 python3-tk나 DISPLAY 설정은 필요 없습니다. 가상환경 사용법은 Python 공식 문서
  (https://docs.python.org/3/using/windows.html)를 참고하면 됩니다.

  ## 2. Teensy와 IMU 펌웨어 설치

  1. Arduino IDE (https://docs.arduino.cc/software) 데스크톱 버전 설치
  2. Teensyduino (https://www.pjrc.com/teensy/td_download.html) 설치
  3. Arduino IDE의 Library Manager에서 다음 라이브러리 설치:

  SparkFun BNO080 Arduino Library

  또는 SparkFun 공식 라이브러리 (https://github.com/sparkfun/SparkFun_BNO080_Arduino_Library)를 사용합니다.

  Arduino IDE에서:

  imu.ino 열기
  도구 → 보드 → 사용하는 Teensy 선택
  도구 → 포트 → Teensy COM 포트 선택
  Upload

  현재 펌웨어 핀 설정:

  SDA = Teensy 17번 핀
  SCL = Teensy 16번 핀
  I2C 주소 = 0x4B

  IMU와 Teensy의 GND도 연결해야 합니다.

  Teensyduino 설치 시에는 Microsoft Store판 Arduino IDE보다 데스크톱/일반 설치판을 사용하는 것이 안전합니다.

  ## 3. Dynamixel 모터 연결

  Windows PC에 Dynamixel USB 어댑터 또는 U2D2를 연결합니다.

  map.py 실행 후:

  Dynamixel motor
  → Port Search
  → 검색된 COM 포트 선택
  → Connect

  모터 설정은 현재 다음과 같습니다.

  Dynamixel ID = 3

  Teensy IMU는 별도의 COM 포트로 선택합니다.

  Teensy IMU
  → Port Search
  → IMU COM 포트 선택
  → Baud rate = 115200
  → Connect

  모터 포트와 IMU 포트는 서로 다른 COM 포트일 수 있습니다.

  ## 4. 설치 확인

  python -c "import cv2, glfw, imgui, OpenGL, serial, dynamixel_sdk, openpyxl; print('Python dependencies OK')"

  정상이라면:

  Python dependencies OK

  가상환경을 다음에 다시 사용할 때는 프로젝트 폴더에서:

  .\.venv\Scripts\Activate.ps1