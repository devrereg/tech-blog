---
title: "T2 맥북 프로 + Ubuntu 24.04: Wi-Fi 연결부터 Chrome 설치까지"
categories: [ROS2]
tags: [ubuntu, macbook-t2, wifi, broadcom, nmcli]
description: "T2 칩 MacBook에 설치한 Ubuntu 24.04에서 Broadcom Wi-Fi가 안 잡히는 문제를 드라이버 재로딩과 feature_disable 옵션으로 해결하고 Chrome까지 설치하는 과정."
---

T2 칩이 들어간 MacBook Pro(터치바 모델)에 Ubuntu 24.04를 설치한 뒤, Wi-Fi가 안 잡히는 문제를 해결하고 Chrome을 설치하는 과정을 정리한 튜토리얼입니다.

## 환경

- 기기: MacBook Pro (T2 칩, 터치바 모델)
- OS: Ubuntu 24.04 (t2-ubuntu 커널/저장소 사용)
- 무선랜: Broadcom 칩 (`brcmfmac` 드라이버, 장치 이름 `wlp1s0f0`)

## 증상

- 오른쪽 위 메뉴에 Wi-Fi가 안 보이거나 목록이 비어 있음
- `sudo apt update` 시 `Temporary failure resolving ...` 오류 (인터넷 없음)
- `nmcli device wifi list` 결과가 헤더만 나오고 비어 있음
- 목록이 보여도 연결 시 `The Wi-Fi network could not be found` 또는 비밀번호 창이 반복해서 뜸

---

## 1단계: 무선랜 장치가 인식됐는지 확인

```bash
nmcli device status
```

`wlp1s0f0` 가 `wifi` 타입으로 보이면 장치는 인식된 상태입니다.

> 참고: `t2_ncm` 은 맥북 내부 T2 칩 통신용 장치라 인터넷과 상관없으니 무시해도 됩니다.

## 2단계: 드라이버 로그 확인

```bash
sudo dmesg | grep -i brcmf
```

아래와 같은 오류가 반복되면 **무선랜 칩이 멈춘(먹통) 상태**입니다. 펌웨어가 없는 게 아니라, 칩이 명령에 응답하지 않는 것입니다.

```text
brcmf_msgbuf_tx_ioctl: Failed to reserve space in commonring
brcmf_cfg80211_get_tx_power: error (-12)
```

## 3단계: 드라이버 다시 불러오기 (칩 리셋)

```bash
sudo modprobe -r brcmfmac_wcc
sudo modprobe -r brcmfmac
sudo modprobe brcmfmac
sleep 10
nmcli device wifi list
```

주변 와이파이 목록이 나오면 성공입니다.

> ⚠️ **오타 주의:** 세 번째 줄을 잘못 입력하면(예: `modporbe`) 드라이버가 빠진 채로 남아서 무선랜 장치 자체가 사라집니다. 이럴 땐 `sudo modprobe brcmfmac` 만 다시 입력하면 됩니다. 키보드 **↑** 키로 이전 명령을 불러오면 오타를 줄일 수 있습니다.
>
> `modprobe -r brcmfmac` 실행 시 "module is in use" 오류가 나면 NetworkManager가 드라이버를 붙잡고 있는 상태입니다. `sudo systemctl stop NetworkManager` 로 멈춘 뒤 다시 시도하고, 끝나면 `sudo systemctl start NetworkManager` 로 복구하면 됩니다.

## 4단계: 드라이버 옵션 추가 (연결 실패 해결)

목록은 보이는데 연결이 계속 실패하거나 비밀번호 창이 반복해서 뜨면, 문제가 되는 일부 기능을 끄는 옵션을 추가해 해결할 수 있습니다. 이 환경(Ubuntu 24.04)에서는 이 옵션으로 해결됐습니다. 다만 t2linux 공식 가이드는 이 옵션을 "T2 맥 공통 문제"가 아니라 **wpa_supplicant 2.11 이후 리그레션으로 인한 Arch/EndeavourOS 한정 이슈**로 설명하고 있으니, 다른 배포판이라면 증상이 다를 수 있다는 점을 참고하세요. (출처: [t2linux wiki](https://wiki.t2linux.org/guides/wifi-bluetooth/))

```bash
echo "options brcmfmac feature_disable=0x82000" | sudo tee /etc/modprobe.d/brcmfmac.conf
```

> 참고: `tee` 는 기존 파일이 있으면 덮어씁니다(이어쓰기가 아님). 나중에 옵션을 추가/수정할 때는 파일 내용을 직접 편집하세요.

적용 여부 확인:

```bash
modprobe -c | grep brcmfmac
```

결과에 `options brcmfmac feature_disable=0x82000` 줄이 포함되어 있으면 정상입니다. (`/etc`, `/run`, `/lib` 등 여러 디렉터리 설정이 병합되어 출력되므로 반드시 첫 줄은 아닐 수 있습니다. 함께 보이는 `alias` 줄들은 지원 칩 목록이라 무시)

> 참고: 이 옵션은 `/sys/module/brcmfmac/parameters/` 경로에 파일로 나타나지 않으므로, `cat` 으로 확인하면 `No such file or directory` 가 나오는 게 정상입니다.

옵션 추가 후에는 **재시작이 아닌 완전 종료 후 다시 켜기**를 합니다.

```bash
sudo poweroff
```

5~10초 기다린 뒤 전원 버튼으로 켭니다.

## 5단계: Wi-Fi 연결

이전에 실패한 연결 정보가 남아 있으면 먼저 지웁니다. ("not found" 가 나와도 괜찮습니다.)

```bash
nmcli connection delete "와이파이이름"
```

드라이버를 다시 불러온 직후, 목록이 보일 때 **바로** 연결합니다.

```bash
nmcli device wifi rescan
sleep 5
nmcli --ask device wifi connect "와이파이이름"
```

`Password:` 가 나오면 비밀번호를 입력합니다. 아래 메시지가 나오면 성공입니다.

```text
Device 'wlp1s0f0' successfully activated with '...'
```

### 비밀번호 입력 팁

- **한/영 상태 확인:** 한글 모드면 영문이 한글 자모로 들어갑니다.
- **헷갈리는 글자:** 대문자 `I` / 소문자 `l`, 숫자 `0` / 대문자 `O`
- 와이파이 비밀번호는 **최소 8자리**입니다. 짧게 입력되면 `802-11-wireless-security.psk: property is invalid` 오류가 납니다.
- GUI 비밀번호 창에서는 **눈 모양 아이콘**으로 입력 내용을 확인할 수 있습니다.

## 6단계: 절전 기능 끄기

이 칩은 절전 모드에서 자주 멈추기 때문에 꺼 두는 것이 좋습니다. 아래 방법은 설정이 저장되어 재부팅 후에도 유지됩니다.

```bash
nmcli connection modify "와이파이이름" 802-11-wireless.powersave 2
nmcli connection up "와이파이이름"
```

`2` 는 "절전 끄기"라는 뜻입니다.

## 7단계: 인터넷 확인

```bash
sudo apt update
```

`Temporary failure resolving` 대신 `Get:1 http://...` 같은 줄이 나오면 인터넷이 정상입니다.

> 참고: 최소 설치 버전에는 `ping`, `iw` 가 없을 수 있습니다. 필요하면 설치하세요.
> ```bash
> sudo apt install iputils-ping iw
> ```

## 8단계: Chrome 설치

```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb
rm google-chrome-stable_current_amd64.deb
```

- `./` 를 꼭 붙여야 현재 폴더의 파일로 인식합니다.
- 설치 시 구글 저장소가 자동 추가되어, 이후 `sudo apt upgrade` 로 Chrome도 함께 업데이트됩니다.
- 실행: 프로그램 메뉴에서 "Google Chrome" 검색 또는 터미널에서 `google-chrome`

---

## 문제 해결 요약

| 증상 | 원인 | 해결 |
|---|---|---|
| `nmcli device wifi list` 가 비어 있음 | 무선랜 칩 멈춤 | 3단계 드라이버 재로딩 |
| 목록은 보이는데 `The Wi-Fi network could not be found` | 검색 후 칩이 다시 멈춤 | 드라이버 재로딩 직후 바로 연결 |
| 비밀번호 창이 반복해서 뜸 | 비밀번호 오입력 또는 드라이버 문제 | 비밀번호 확인 → 4단계 옵션 추가 |
| `No network with SSID ... found` | 칩 멈춤 | 3단계 드라이버 재로딩 |
| 목록에 아무것도 안 나옴 (헤더도 없음) | 드라이버가 빠진 상태 | `sudo modprobe brcmfmac` |
| `ping` / `iw`: command not found | 최소 설치라 미포함 | `sudo apt install iputils-ping iw` |

## 재부팅 후 Wi-Fi가 안 잡힐 때

한 번 연결한 와이파이는 저장되어 자동으로 연결됩니다. 재부팅이나 절전 해제 후 와이파이가 안 잡히면 3단계 드라이버 재로딩만 다시 하면 됩니다.

## 참고 자료

- t2linux Wi-Fi/Bluetooth 가이드: https://wiki.t2linux.org/guides/wifi-bluetooth/
