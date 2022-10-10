AA = "AA"
AE = "AE"
AH = "AH"
AO = "AO"
AW = "AW"
AX = "AX"  # not in MFA
AXR = "AXR"  # not in MFA
AY = "AY"
EH = "EH"
ER = "ER"
EY = "EY"
IH = "IH"
IX = "IX"  # not in MFA
IY = "IY"
OW = "OW"
OY = "OY"
UH = "UH"
UW = "UW"
UX = "UX"  # not in MFA

B = "B"
CH = "CH"
D = "D"
DH = "DH"
DX = "DX"  # not in MFA
EL = "EL"  # not in MFA
EM = "EM"  # not in MFA
EN = "EN"  # not in MFA
F = "F"
G = "G"
HH = "HH"
H = "H"  # not in MFA
JH = "JH"
K = "K"
L = "L"
M = "M"
N = "N"
NG = "NG"
NX = "NX"  # not in MFA
P = "P"
Q = "Q"  # not in MFA
R = "R"
S = "S"
SH = "SH"
T = "T"
TH = "TH"
V = "V"
W = "W"
WH = "WH"  # not in MFA
Y = "Y"
Z = "Z"
ZH = "ZH"


VOWELS = {
  AA,
  AE,
  AH,
  AO,
  AW,
  AX,
  AXR,
  AY,
  EH,
  ER,
  EY,
  IH,
  IX,
  IY,
  OW,
  OY,
  UH,
  UW,
  UX,
}

STRESS_NONE = "0"
STRESS_NONE_ALT = ""
STRESS_PRIMARY = "1"
STRESS_SECONDARY = "2"

STRESS_MARKERS = {STRESS_NONE, STRESS_NONE_ALT, STRESS_PRIMARY, STRESS_SECONDARY}

VOWELS_WITH_NUMBERED_STRESSES = {
    f"{vowel}{stress_nr}"
    for vowel in VOWELS
    for stress_nr in [STRESS_NONE, STRESS_PRIMARY, STRESS_SECONDARY]
}

VOWELS_WITH_STRESSES = {
    f"{vowel}{stress_nr}" for vowel in VOWELS for stress_nr in STRESS_MARKERS}

CONSONANTS = {
  B,
  CH,
  D,
  DH,
  DX,
  EL,
  EM,
  EN,
  F,
  G,
  HH,
  H,
  JH,
  K,
  L,
  M,
  N,
  NG,
  NX,
  P,
  Q,
  R,
  S,
  SH,
  T,
  TH,
  V,
  W,
  WH,
  Y,
  Z,
  ZH,
}

ALL_ARPA_EXCL_STRESSES = VOWELS | CONSONANTS
ALL_ARPA_INCL_STRESSES = VOWELS_WITH_STRESSES | CONSONANTS

# print("\n".join(sorted(ALL_ARPA_INCL_STRESSES)))
