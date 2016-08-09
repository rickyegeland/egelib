"""
Conversions and functions specific to an instrument.
"""

def tycho2johnson(BT, VT, lum='V'):
    BmVT = BT-VT
    if lum == 'V' and BmVT > 0.50:
        # TODO: other validity ranges
        # ESA 1997 Sec 1.3 Appendix 4 Equation 1.3.26 valid for BT-VT > 0.50 lum class V
        z = BmVT - 0.90
        BmV = BmVT - 0.115 - 0.229 * z + 0.043*z**3
    elif lum == 'III' and BmVT > 0.65 and BmVT < 1.10:
        #  ESA 1997 Sec 1.3 Appendix 4 Equation 1.3.28
        z = BmVT - 0.22
        BmV = BmVT - 0.113 - 0.258*z + 0.40*z**3
    #  ESA 1997 Sec 1.3 Appendix 4 Equation 1.3.33
    V = VT + 0.0036 - 0.1284*BmVT + 0.0442*BmVT**2 - 0.015*BmVT**3
    B = BmV + V    
    return B, V
