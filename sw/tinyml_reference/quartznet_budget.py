# QuartzNet 15x5 exact param / MAC-per-output-frame budget (from NeMo v1.23 config)
# separable=true everywhere except C3 (kernel=1) ; C1 stride 2 ; n_mels=64 ; 100 fps in -> 50 fps after C1
blocks = [  # (name, C_out, K, R, residual, separable)
 ("C1", 256, 33, 1, False, True),
 *[(f"B1.{i}",256,33,5,True,True) for i in range(3)],
 *[(f"B2.{i}",256,39,5,True,True) for i in range(3)],
 *[(f"B3.{i}",512,51,5,True,True) for i in range(3)],
 *[(f"B4.{i}",512,63,5,True,True) for i in range(3)],
 *[(f"B5.{i}",512,75,5,True,True) for i in range(3)],
 ("C2", 512, 87, 1, False, True),
 ("C3", 1024, 1, 1, False, False),
 ("C4(dec)", 28, 1, 1, False, False),
]
cin = 64; tot_p = tot_dw = tot_pw = tot_res = 0
print(f"{'block':>9} {'Cin':>5} {'Cout':>5} {'K':>3} {'R':>2} {'dw_MAC':>10} {'pw_MAC':>12} {'res_MAC':>9} {'total':>12}")
for name, cout, K, R, res, sep in blocks:
    dw = pw = 0
    c = cin
    for r in range(R):
        if sep: dw += c*K              # depthwise: C_in taps of length K
        else:   pw += c*cout*K; c = cout; continue
        pw += c*cout; c = cout
    rmac = cin*cout if res else 0      # 1x1 residual projection
    tot_dw+=dw; tot_pw+=pw; tot_res+=rmac
    t = dw+pw+rmac; tot_p += t
    print(f"{name:>9} {cin:>5} {cout:>5} {K:>3} {R:>2} {dw:>10,} {pw:>12,} {rmac:>9,} {t:>12,}")
    cin = cout
print(f"\ndepthwise {tot_dw:,} ({100*tot_dw/tot_p:.1f}%)  pointwise {tot_pw:,} ({100*tot_pw/tot_p:.1f}%)  residual {tot_res:,} ({100*tot_res/tot_p:.1f}%)")
print(f"TOTAL weights (= MACs per output frame): {tot_p:,}  ({tot_p/1e6:.2f} M)")
FPS=50
print(f"\nreal-time @ {FPS} output-frames/s : {tot_p*FPS/1e6:,.0f} MMAC/s  = {2*tot_p*FPS/1e9:.2f} GOP/s")
print(f"int8 weight footprint: {tot_p/1e6:.1f} MB   (repo's whole address space today: 0.256 MB)")
for L,f in [(4,269e6),(16,269e6),(64,269e6),(256,269e6),(64,500e6),(256,500e6)]:
    thr=L*f; print(f"  LANES={L:>4} @ {f/1e6:.0f}MHz -> {thr/1e6:,.0f} MMAC/s = {thr/(tot_p*FPS):.2f}x real-time (at 100% util)")
# activation working set
print("\nactivation buffer per chunk of T frames (int8, 512ch): T=100 -> %.0f KB/tensor" % (512*100/1024))

def build(S, R, cs, ks, cenc=1024, nmels=64, vocab=28):
    bl=[("C1",cs[0],ks[0],1,False,True)]
    for i,(c,k) in enumerate(zip(cs,ks)):
        bl += [(f"B{i}.{j}",c,k,R,True,True) for j in range(S)]
    bl += [("C2",cs[-1],87,1,False,True),("C3",cenc,1,1,False,False),("C4",vocab,1,1,False,False)]
    cin=nmels; tot=0
    for name,cout,K,Rr,res,sep in bl:
        dw=pw=0; c=cin
        for r in range(Rr):
            if sep: dw+=c*K; pw+=c*cout; c=cout
            else: pw+=c*cout*K; c=cout
        tot += dw+pw+(cin*cout if res else 0); cin=cout
    return tot

CS=[256,256,512,512,512]; KS=[33,39,51,63,75]
print("\n=== variants (params == MACs/output-frame) ===")
for label,S,R,cs,ks,ce in [
  ("QuartzNet 15x5 (paper)",3,5,CS,KS,1024),
  ("QuartzNet 10x5",2,5,CS,KS,1024),
  ("QuartzNet 5x5",1,5,CS,KS,1024),
  ("QN 5x5, C=256 flat",1,5,[256]*5,KS,512),
  ("QN 5x5, C=128 flat",1,5,[128]*5,KS,512),
  ("QN 5x3, C=128 flat",1,3,[128]*5,KS,256),
  ("QN 3x3, C=128",1,3,[128]*3,KS[:3],256),
]:
    p=build(S,R,cs,ks,ce)
    print(f"{label:<24} {p/1e6:6.2f} M params = {p/1e6:6.2f} MB int8 | real-time {p*50/1e6:6.0f} MMAC/s | fits 512KB SRAM? {'YES' if p<=512*1024 else 'no'}")
