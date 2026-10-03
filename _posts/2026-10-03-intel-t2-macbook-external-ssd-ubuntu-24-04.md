---
title: "Intel(T2) MacBook에 외장 SSD로 Ubuntu 24.04 설치하기"
date: 2026-10-03 17:00:00 +0900
categories: [ROS2]
tags: [ubuntu, macbook-t2, dual-boot, grub, external-ssd]
description: "T2 칩 Intel MacBook에서 macOS는 그대로 두고 외장 SSD에 Ubuntu 24.04를 설치해 Option 키로 골라 부팅하는 전 과정 정리."
---

T2 칩이 있는 Intel MacBook에서 macOS는 그대로 두고, 외장 SSD에 Ubuntu 24.04를 설치해 Option 키로 골라 부팅하는 방법입니다. 내장 디스크는 파티션을 나누지 않습니다.

| 항목 | 이 문서의 기준 |
| --- | --- |
| Mac | MacBook Pro 13" 2020 (Intel Core i7, T2 칩, Thunderbolt 3 포트 2개) |
| macOS | Sequoia 15.8 |
| 설치 대상 | SK hynix X31 1TB 외장 SSD |
| Ubuntu | t2linux의 T2용 Ubuntu 24.04 (커널 7.1.8-t2-noble) |

전체 흐름은 다음 9단계입니다. ROKEY BOOT CAMP의 Windows용 설치 가이드와 순서는 같고, Mac에서 달라지는 부분(Rufus 대신 dd, BIOS 대신 Option 키, T2 보안 설정, 부트로더 수정)을 채웠습니다.

1. Mac 확인 (Intel, T2)
2. T2용 Ubuntu ISO 다운로드와 검증
3. 부팅 USB 만들기
4. 시동 보안 설정
5. USB로 부팅
6. 라이브 환경에서 SSD 확인
7. 설치 프로그램 진행
8. 부트로더 위치 확인과 수정
9. 첫 부팅과 영구 수정

가장 중요한 주의점은 두 가지입니다. 설치 화면에서 **Erase disk를 절대 고르지 않는 것**, 그리고 설치 직후 **부팅 파일이 SSD에 들어갔는지 확인하는 것**입니다. 이 설치 프로그램은 부트로더 장치를 SSD로 지정해도 Mac 내장 디스크에 부팅 파일을 넣는 경우가 있습니다.

## 준비물

장치는 **USB 메모리 1개와 외장 SSD 1개**, 두 가지가 필요합니다. USB는 설치 프로그램을 띄우는 용도이고, SSD에 Ubuntu가 실제로 설치됩니다. SSD 하나로 두 역할을 동시에 할 수는 없습니다.

| 준비물 | 조건 | 비고 |
| --- | --- | --- |
| USB 메모리 | 8GB 이상 | 내용이 모두 지워짐 (예: SanDisk Cruzer Blade 32GB) |
| 외장 SSD | 256GB 이상 권장 | 내용이 모두 지워짐 (예: X31 1TB) |
| USB-C 허브 | USB-A 포트가 있는 것 | Thunderbolt 포트가 2개뿐이라 필요 |
| 충전기 | — | 권장. 없으면 배터리 80% 이상으로 진행 |
| 휴대폰 | — | 재부팅 중 안내를 보거나 화면을 찍을 때 사용 |

포트 연결은 이렇게 나눕니다.

- 맥북 포트 1: 허브, 그리고 허브에 USB 메모리
- 맥북 포트 2: 외장 SSD를 USB-C 케이블로 직접 연결

시작 전에 USB와 SSD에 있는 파일, 그리고 Mac의 중요한 파일을 백업해 두세요.

## 1단계: Mac 확인하기

Intel 칩이고 T2 칩이 있는 Mac이면 이 문서대로 진행합니다. Apple Silicon(M1 이상)은 이 방법을 쓸 수 없습니다.

**Intel 칩 확인**

1. Apple 메뉴 → **이 Mac에 관하여**를 엽니다.
2. "프로세서" 항목에 **Intel**이 적혀 있는지 봅니다.

**T2 칩 확인**

1. "이 Mac에 관하여" → **추가 정보…** → 시스템 리포트…를 엽니다.
2. 왼쪽에서 **하드웨어 → 컨트롤러**를 선택합니다.
3. 모델 이름에 **Apple T2 보안 칩**이 보이면 T2 Mac입니다.

터미널로 확인해도 됩니다. 결과에 `Apple T2 Security Chip`이 나오면 T2 Mac입니다.

```bash
system_profiler SPiBridgeDataType
```

2018~2020년 Intel MacBook Pro는 모두 T2 칩이 있습니다. T2 Mac은 일반 Ubuntu ISO로는 내장 키보드, 트랙패드, Wi-Fi가 동작하지 않아서 T2용 ISO가 필요합니다.

## 2단계: T2용 Ubuntu ISO 받기

ubuntu.com의 공식 ISO가 아니라 [t2linux의 T2-Ubuntu 릴리스](https://github.com/t2linux/T2-Ubuntu/releases/latest)에서 받습니다. ISO는 여러 조각으로 나뉘어 있어서 합친 뒤 검증합니다.

**받을 파일 (5개)**

| 파일 | 크기 | 용도 |
| --- | --- | --- |
| `sha256-ubuntu-24.04` | 104 Bytes | 검증용 체크섬 |
| `ubuntu-24.04-7.1.8-t2-noble.iso.00` | 700 MB | ISO 조각 1 |
| `ubuntu-24.04-7.1.8-t2-noble.iso.01` | 700 MB | ISO 조각 2 |
| `ubuntu-24.04-7.1.8-t2-noble.iso.02` | 700 MB | ISO 조각 3 |
| `ubuntu-24.04-7.1.8-t2-noble.iso.03` | 247 MB | ISO 조각 4 |

kubuntu, xubuntu, unity, budgie, cinnamon이 들어간 파일과 26.04 파일은 받지 않습니다. 파일 이름의 버전 번호(7.1.8)는 릴리스에 따라 다를 수 있으니, 아래 명령의 이름도 실제 받은 파일에 맞춰 바꾸세요. 체크섬 파일은 클릭하면 브라우저에 글자만 뜰 수 있는데, 그럴 때는 오른쪽 클릭 → "링크된 파일 다운로드"를 고릅니다.

**조각 합치기와 검증 (Mac 터미널)**

```bash
cd ~/Downloads
cat ubuntu-24.04-7.1.8-t2-noble.iso.0* > ubuntu-24.04-7.1.8-t2-noble.iso
shasum -a 256 ubuntu-24.04-7.1.8-t2-noble.iso
cat sha256-ubuntu-24.04
```

마지막 두 명령이 출력한 64자리 값이 같으면 성공입니다. 다르면 조각 중 하나가 잘못 받아진 것이니 다시 받습니다. 합친 `.iso`(약 2.3GB)는 보관하고, 조각 파일은 지워도 됩니다.

## 3단계: 부팅 USB 만들기

Mac에서는 Rufus 대신 터미널의 `dd` 명령으로 ISO를 USB에 굽습니다. 디스크 번호를 잘못 넣으면 그 디스크가 지워지므로, **외장 SSD는 빼고 USB만 꽂은 상태**로 진행합니다.

1. 허브에 USB만 꽂고 디스크 목록을 봅니다.

   ```bash
   diskutil list
   ```
2. `(external, physical)`이고 용량이 USB와 비슷한 디스크를 찾습니다. 예: `/dev/disk2`, 30.8 GB. `disk0`(internal)과 `(synthesized)`가 붙은 디스크는 macOS용이라 건드리지 않습니다.
3. 아래 명령의 `disk2`를 본인 USB 번호로 바꿔 실행합니다.

   ```bash
   diskutil unmountDisk /dev/disk2
   sudo dd if=$HOME/Downloads/ubuntu-24.04-7.1.8-t2-noble.iso of=/dev/rdisk2 bs=1m
   ```
4. `Password:`가 나오면 Mac 로그인 비밀번호를 입력합니다. 입력해도 화면에 보이지 않는 게 정상입니다.
5. 3~4분 기다리면 `bytes transferred`가 나오며 끝납니다. 진행 상황은 control + T로 볼 수 있습니다.
6. "디스크를 읽을 수 없습니다" 창이 뜨면 **꺼내기**를 누릅니다. 정상입니다.

참고 결과: `2461433856 bytes transferred in 201 secs`

## 4단계: 시동 보안 설정 바꾸기

T2 Mac은 기본 설정으로는 외장 디스크와 Linux로 부팅할 수 없어서, 복구 모드에서 보안 설정 두 가지를 바꿉니다. 재시동이 필요하니 이 문서를 휴대폰으로 열어 두세요.

1. Apple 메뉴 → **시스템 종료**로 Mac을 끕니다.
2. 전원을 켜자마자 **⌘ Command + R**을 계속 누르고, Apple 로고가 나오면 손을 뗍니다.
3. "암호를 알고 있는 사용자 선택" 화면에서 본인 계정을 고르고 **다음**을 누른 뒤 Mac 비밀번호를 입력합니다.
4. macOS 유틸리티 창의 항목은 누르지 말고, 화면 맨 위 메뉴에서 **유틸리티 → 시동 보안 유틸리티**를 엽니다.
5. 두 가지를 바꿉니다. 비밀번호를 물으면 입력합니다.

| 항목 | 기본값 | 바꿀 값 |
| --- | --- | --- |
| 보안 시동 | 완전 보안 | **보안 없음** |
| 허용된 시동 미디어 | 외부 또는 제거 가능한 미디어에서 시동 허용 안 함 | **외부 또는 제거 가능한 미디어에서 시동 허용** |

"펌웨어 암호 켜기…"는 누르지 않습니다. 창을 닫고 Apple 메뉴 → **시스템 종료**를 합니다. 이 설정은 Ubuntu를 쓰는 동안 계속 유지해야 합니다.

## 5단계: USB로 부팅하기

Mac에는 BIOS 대신 **Option 키 시동 관리자**가 있습니다. 여기서 노란색 **EFI Boot**를 골라 USB로 부팅합니다.

1. 부팅 USB(허브)와 외장 SSD를 꽂습니다. SSD를 늦게 꽂으면 Ubuntu에서 인식이 안 될 수 있으니 이때 같이 꽂아 둡니다.
2. **⌥ Option 키를 먼저 누른 상태**에서 전원 버튼을 누르고, 디스크 아이콘이 나올 때까지 계속 누릅니다.
3. EFI Boot 아이콘을 **마우스로 클릭**해 선택 테두리를 옮기고, 그 아래 ↑ 화살표를 클릭합니다. 그냥 Enter를 누르면 Macintosh HD로 부팅되니 주의합니다.
4. GRUB 메뉴에서 **Try or Install Ubuntu**를 선택합니다.
5. 1~3분 뒤 Ubuntu 라이브 바탕화면이 나옵니다. Touch Bar에 밝기·음량 버튼이 켜지면 T2 드라이버가 잘 동작하는 것입니다.

EFI Boot가 3개 보이는 것은 정상입니다. 모두 같은 USB이고, **맨 오른쪽 것**이 부팅됐습니다. 다른 것을 골라 "소프트웨어를 업데이트해야 합니다" 경고가 나오면 **업데이트**는 누르지 말고 Mac을 끈 뒤 다른 EFI Boot로 다시 시도합니다.

GRUB 메뉴 옵션은 다음과 같습니다.

| 옵션 | 언제 쓰나 |
| --- | --- |
| Try or Install Ubuntu | 기본 |
| Ubuntu (Safe Graphics) | 화면이 깨질 때 |
| Ubuntu (NVMe blacklisted) | 5분 넘게 멈출 때 |

## 6단계: 라이브 환경에서 SSD 확인하기

설치 전에 Ubuntu가 외장 SSD를 인식했는지, 어떤 이름으로 보이는지 확인합니다. **control + option + T**로 터미널을 열고 실행합니다.

```bash
lsblk -d -o NAME,SIZE,MODEL,TRAN
```

| 이름 (예) | 크기 | 모델 | 정체 |
| --- | --- | --- | --- |
| `sda` | 953.9G | PSSD X31 | **여기에 설치** |
| `sdb` | 28.7G | Cruzer Blade | 부팅 USB |
| `nvme0n1` | 233.8G | APPLE SSD AP0256N | Mac 내장 디스크, 건드리지 않음 |
| `loop0` | 2.2G | — | 라이브 시스템, 무시 |

`sda`, `sdb` 같은 이름은 부팅할 때마다 바뀔 수 있습니다. 항상 **크기와 모델 이름**으로 구분하세요. SSD가 목록에 없으면 라이브 Ubuntu를 끄고, SSD를 꽂은 상태로 5단계부터 다시 부팅합니다.

## 7단계: 설치 프로그램 진행하기

바탕화면의 **Install** 아이콘을 더블클릭합니다. T2용 ISO는 PDF 가이드와 모양이 다른 이전 방식의 설치 프로그램을 씁니다. 핵심은 **Something else**로 SSD에만 파티션을 만드는 것입니다.

**화면별 선택**

| 화면 | 선택 |
| --- | --- |
| 언어 / 키보드 | English / English (US) |
| Wireless | I don't want to connect to a Wi-Fi network right now |
| Updates and other software | Normal installation, 두 체크박스 모두 해제 |
| Unmount partitions that are in use? (`/dev/sda`) | **Yes** (연결만 해제, 데이터는 아직 안 지워짐) |
| Installation type | **Something else** (Erase disk 절대 금지) |

Installation type 화면에 "no detected operating systems"라고 나와도 macOS는 그대로 있습니다. Ubuntu가 APFS를 알아보지 못할 뿐이라, 이 상태에서 Erase disk를 고르면 macOS가 지워집니다.

**파티션 만들기 (Something else 화면)**

1. 목록에서 숫자 없는 **`/dev/sda`** 줄을 클릭하고 **New Partition Table…** → **Continue**를 누릅니다. `nvme0n1`으로 시작하는 줄은 건드리지 않습니다.
2. `/dev/sda` 아래 **free space**를 클릭하고 +를 눌러 아래 표대로 하나씩 만듭니다.

| 순서 | Size | Use as | Mount point |
| --- | --- | --- | --- |
| 1 | 1024 MB | EFI System Partition | — |
| 2 | 200000 MB | Ext4 journaling file system | `/` |
| 3 | 나머지 전부 (기본값) | Ext4 journaling file system | `/home` |

3. 맨 아래 **Device for boot loader installation**을 `/dev/nvme0n1 APPLE SSD`에서 `/dev/sda Hynix PSSD X31`으로 바꿉니다.
4. **Install Now**를 누릅니다. swap 경고가 나오면 Continue를 누릅니다.
5. **Write the changes to disks?** 창에 `sda`만 있는지 확인합니다. `nvme0n1`이 하나라도 있으면 Go Back을 누릅니다.

| 정상 확인 창 내용 |
| --- |
| partition #1 of SCSI1 (0,0,0) (sda) as ESP |
| partition #2 of SCSI1 (0,0,0) (sda) as ext4 |
| partition #3 of SCSI1 (0,0,0) (sda) as ext4 |

6. 시간대는 Seoul을 고르고, 계정을 만듭니다. **Require my password to log in**을 선택합니다.
7. 10~30분 기다립니다. 중간에 USB나 SSD를 뽑지 않습니다.
8. 완료 창에서 **Continue Testing**을 누릅니다. 재시동하지 않고 8단계로 갑니다.

## 8단계: 부트로더 위치 확인하고 고치기

이 설치 프로그램은 부트로더 장치를 SSD로 지정해도 **부팅 파일을 Mac 내장 EFI 파티션에 넣었습니다.** 그대로 재시동하면 SSD로 부팅되지 않으니, 라이브 환경에서 SSD로 옮기고 내장 EFI를 원래대로 정리합니다.

명령이 길어서 오타가 나기 쉽습니다. 라이브 Ubuntu에서 Wi-Fi를 연결하고 Firefox로 이 문서를 열어, 복사한 뒤 터미널에 **control + shift + V**로 붙여넣는 것을 권장합니다. 아래 명령은 SSD가 `sda`일 때 기준입니다.

**① 확인: 부팅 파일이 어디에 있나**

```bash
sudo mkdir -p /mnt/x31 /mnt/mac
sudo mount /dev/sda1 /mnt/x31
ls -la /mnt/x31
sudo mount -o ro /dev/nvme0n1p1 /mnt/mac
ls -la /mnt/mac/EFI
```

| 위치 | 문제가 있을 때 | 정상일 때 |
| --- | --- | --- |
| SSD (`/mnt/x31`) | 비어 있음 | `EFI` 폴더 있음 |
| Mac 내장 (`/mnt/mac/EFI`) | `APPLE`, `BOOT`, `ubuntu` | `APPLE`만 |

내장 EFI의 `BOOT`, `ubuntu` 폴더 날짜가 설치한 시각과 같으면 설치 프로그램이 만든 것입니다. 둘 다 정상이면 ②~④를 건너뛰고 9단계로 갑니다.

**② 설치된 Ubuntu 안으로 들어가기**

```bash
sudo umount /mnt/x31 /mnt/mac
sudo mkdir -p /mnt/root
sudo mount /dev/sda2 /mnt/root
sudo mount /dev/sda1 /mnt/root/boot/efi
for d in dev proc sys run; do sudo mount --bind /$d /mnt/root/$d; done
sudo chroot /mnt/root
```

프롬프트가 `root@ubuntu:/#`로 바뀌면 됩니다.

**③ SSD에 부팅 파일 설치, EFI 설정 수정**

```bash
grub-install --target=x86_64-efi --efi-directory=/boot/efi --removable --no-nvram
U=$(blkid -s UUID -o value /dev/sda1)
echo $U
sed -i "/boot\/efi/s/UUID=[^ ]*/UUID=$U/" /etc/fstab
update-grub
ls /boot/efi/EFI
grep efi /etc/fstab
exit
sudo umount -R /mnt/root
```

- `grub-install`이 Installation finished. No error reported.를 출력해야 합니다.
- `ls` 결과에 `BOOT`가 보이고, `grep` 결과의 UUID가 `echo $U` 값과 같아야 합니다.
- `--removable`은 Option 메뉴에 EFI Boot로 보이게 하고, `--no-nvram`은 Mac 부팅 설정을 건드리지 않게 합니다.
- `update-grub`의 `cannot find a GRUB drive for /dev/sdb1` 오류는 부팅 USB 때문이라 무시합니다.
- fstab의 `# /boot/efi was on /dev/nvme0n1p1` 줄은 주석이라 그대로 둡니다.

**④ Mac 내장 EFI 정리**

```bash
sudo mount /dev/nvme0n1p1 /mnt/mac
sudo rm -r /mnt/mac/EFI/ubuntu /mnt/mac/EFI/BOOT
ls /mnt/mac/EFI
sudo umount /mnt/mac
```

`ls` 결과에 **`APPLE`만** 남아야 합니다. `APPLE` 폴더는 절대 지우지 않습니다.

## 9단계: 첫 부팅과 영구 수정

`--removable`로만 설치하면 GRUB이 `EFI/ubuntu` 폴더에서 설정 파일을 찾지 못해 `grub>` 명령 화면에서 멈춥니다. 처음 한 번은 수동으로 부팅하고, 로그인한 뒤 영구적으로 고칩니다.

**① SSD로 부팅**

1. 라이브 Ubuntu를 Power Off로 끕니다. "Please remove the installation medium" 메시지가 나오면 **부팅 USB를 뽑고** Enter를 누릅니다. 이미 Enter를 눌렀다면 꺼진 뒤 뽑아도 됩니다.
2. SSD만 꽂은 상태로 **Option 키**를 누른 채 켜고, **EFI Boot**를 선택합니다.

**② `grub>` 화면이 나오면 수동 부팅**

```text
search --file --set=root /boot/grub/grub.cfg
configfile /boot/grub/grub.cfg
```

GRUB 메뉴가 나오면 **Ubuntu**를 선택하고 로그인합니다. 환영 화면은 Ubuntu Pro는 Skip for now, 데이터 공유는 No를 고르고 마무리합니다. "Incomplete Language Support" 창은 Close를 누릅니다.

**③ 영구 수정 (설치된 Ubuntu 터미널)**

```bash
findmnt /boot/efi
```

SOURCE가 `/dev/sda1`이어야 합니다. `nvme0n1p1`이 나오면 멈추고 8단계를 다시 확인합니다. 맞으면 이어서 실행합니다.

```bash
sudo grub-install --target=x86_64-efi --efi-directory=/boot/efi --bootloader-id=ubuntu --no-nvram
sudo update-grub
ls /boot/efi/EFI
```

`ls` 결과에 `BOOT`와 `ubuntu`가 함께 보이면 완료입니다.

**④ 재부팅 테스트**

Restart 후 Option 키 → EFI Boot를 선택했을 때 `grub>` 화면 없이 GRUB 메뉴가 바로 나오면 설치가 끝난 것입니다.

## 평소 사용법

macOS는 그냥 켜면 되고, Ubuntu는 SSD를 꽂고 Option 키로 고릅니다.

| 상황 | 방법 |
| --- | --- |
| Ubuntu 쓰기 | SSD 꽂기 → Option 키를 누른 채 켜기 → EFI Boot 선택 |
| macOS 쓰기 | 그냥 켜기 (SSD는 빼 둬도 됨) |
| SSD 분리 | 반드시 Ubuntu를 종료한 뒤 뽑기 |
| 시동 보안 설정 | 보안 없음 + 외부 미디어 허용을 계속 유지 |

설치 후에 하면 좋은 것:

- 업데이트: Wi-Fi 연결 후 `sudo apt update && sudo apt upgrade -y`
- VS Code: ROKEY 가이드 4장과 같은 방법 (`.deb` 받아서 `sudo dpkg -i code_…deb`)
- 한글 입력: 설정 → Language Support에서 언어 팩 설치 후 한국어 입력기 추가

## 문제 해결

실제 설치 중 겪은 문제와 해결 방법입니다.

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| 허브나 충전기를 꽂자 Mac이 혼자 켜져 macOS 로그인 화면으로 감 | 전원 연결 시 자동 켜짐 기능 | 로그인 화면에서 시스템 종료 → Option 키를 **먼저** 누른 채 전원 켜기 |
| "이 시동 디스크를 사용하려면 소프트웨어를 업데이트해야 합니다" | Macintosh HD가 선택됐거나, 해당 EFI Boot가 맞지 않음 | 업데이트 누르지 않기. 끄고 다른 EFI Boot(맨 오른쪽)로 다시 시도. 계속되면 4단계 보안 설정 재확인 |
| "시동 디스크 선택" 창에 Macintosh HD만 보임 | 복구 모드의 시동 디스크 앱이라 USB가 안 보임 | 창 닫고 시스템 종료 → Option 키로 다시 켜기 |
| EFI Boot가 3개 보임 | 하나의 USB가 여러 번 표시됨 | 정상. 맨 오른쪽부터 시도 |
| EFI Boot가 하나도 안 보임 | USB 연결 불량 | 그 화면에서 허브와 USB를 다시 꽂고 10초 대기. 안 되면 포트 바꿔서 재부팅 |
| 라이브 Ubuntu에서 SSD가 `lsblk`에 안 보임 | 부팅 후 꽂은 장치를 인식 못 함 | SSD를 꽂은 상태로 다시 부팅 |
| 설치 후 SSD의 EFI가 비어 있고 내장 EFI에 `ubuntu` 폴더가 생김 | 설치 프로그램이 부트로더 장치 설정을 무시 | 8단계 ②~④ |
| SSD로 부팅하니 `grub>` 화면에서 멈춤 | `EFI/ubuntu`에 설정 파일이 없음 | 9단계 ②로 부팅 후 ③ 실행 |
| `special device /devnvme0n1p1 does not exist` | 오타 (`/dev` 뒤 슬래시 누락) | `/dev/nvme0n1p1`로 다시 입력 |
| `sed: unterminated 's' command` | 오타 (슬래시 누락) | 8단계 ③의 명령을 복사해 붙여넣기 |

오타를 줄이려면 라이브 Ubuntu에서 Wi-Fi를 연결하고 Firefox로 이 문서를 열어 명령을 복사·붙여넣기 하세요. 터미널 붙여넣기는 control + shift + V입니다.

## 참고 자료

- [t2linux wiki: Pre-Install](https://wiki.t2linux.org/guides/preinstall/)
- [t2linux wiki: Ubuntu Installation](https://wiki.t2linux.org/distributions/ubuntu/installation/)
- [T2-Ubuntu releases](https://github.com/t2linux/T2-Ubuntu/releases/latest)
- [Apple: T2 칩 탑재 Mac의 시동 보안 유틸리티](https://support.apple.com/en-us/102522)
- ROKEY BOOT CAMP, Ubuntu 24.04 LTS Installation Guide (PDF)
