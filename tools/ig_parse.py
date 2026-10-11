import re
p = r'C:\Users\Administrator\AppData\Local\Temp\ig\Intel Intrinsics Guide\files\data.js'
d = open(p, encoding='utf-8', errors='replace').read()
d = d.replace('\\"', '"').replace('\\n', '\n')
# any rcp/rsqrt/exp28-style ER entries? any KNC-ish names?
for probe in ('_mm512_rcp28', '_mm512_exp2a23', '_mm512_rsqrt28', 'vexp2', 'VEXP2',
              'rcp23', 'RCP23', 'exp223', 'instruction name=', '<operation>',
              '_mm512_permutexvar_epi32 ', '_mm512_loadu_ps'):
    print(f"{probe!r}: {d.count(probe)}")
# print one full AVX-512 entry to learn the schema
i = d.find('_mm512_add_ps"')
print('\n--- schema sample ---')
print(d[i-100:i+1200])
