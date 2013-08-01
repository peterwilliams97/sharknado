prog=".\tsp4aj_local20a1.py"
VERSION=30
MAX_CLOSEST=20
MAX_N=30000
RANDOM_SEED=111
DEBUG=False
EPSILON=1e-06
saved_solutions = {'./data/tsp_51_1': (432.41435496853336, [0, 5, 2, 28, 10, 9, 45, 47, 26, 6, 36, 12, 30, 23, 34, 24, 41, 27, 3, 46, 8, 4, 35, 13, 7, 19, 40, 11, 42, 18, 16, 44, 14, 15, 38, 50, 39, 43, 29, 21, 37, 20, 25, 1, 31, 49, 17, 32, 48, 22, 33]), './data/tsp_200_2': (30962.192884879092, [0, 49, 168, 174, 129, 80, 33, 148, 185, 37, 65, 7, 51, 137, 119, 179, 26, 23, 164, 87, 178, 12, 78, 180, 146, 40, 83, 136, 171, 14, 72, 38, 154, 43, 92, 3, 122, 121, 187, 70, 42, 76, 90, 53, 62, 153, 15, 151, 95, 85, 188, 142, 17, 84, 63, 149, 58, 5, 39, 82, 2, 115, 176, 4, 59, 52, 123, 117, 144, 135, 18, 191, 50, 118, 36, 195, 61, 34, 29, 143, 99, 1, 47, 140, 91, 116, 177, 9, 20, 181, 163, 113, 24, 19, 141, 8, 101, 54, 112, 25, 86, 162, 130, 147, 94, 55, 150, 27, 11, 114, 46, 132, 105, 103, 182, 81, 186, 159, 64, 173, 13, 67, 32, 165, 44, 98, 77, 30, 56, 71, 134, 160, 126, 75, 79, 193, 156, 133, 106, 183, 157, 68, 124, 108, 145, 45, 120, 189, 100, 194, 73, 111, 60, 170, 6, 197, 131, 66, 74, 158, 35, 128, 107, 198, 175, 196, 190, 28, 127, 57, 102, 110, 192, 21, 184, 172, 41, 22, 109, 167, 10, 88, 89, 16, 139, 138, 69, 152, 48, 169, 97, 166, 96, 104, 31, 93, 161, 125, 199, 155]), './data/tsp_100_3': (21268.466435012841, [0, 65, 12, 93, 15, 97, 33, 60, 1, 45, 36, 46, 30, 94, 82, 49, 59, 41, 68, 48, 42, 53, 9, 63, 85, 6, 23, 18, 52, 22, 8, 90, 38, 70, 72, 19, 25, 40, 43, 44, 99, 11, 32, 21, 35, 92, 54, 5, 20, 87, 88, 77, 37, 47, 7, 83, 39, 74, 66, 57, 71, 24, 55, 3, 51, 84, 17, 79, 26, 29, 14, 80, 96, 16, 4, 91, 69, 13, 28, 62, 64, 76, 34, 50, 2, 89, 61, 98, 67, 78, 95, 73, 81, 56, 31, 27, 58, 75, 10, 86])}
saved_points = {'./data/tsp_51_1': [(27.0, 68.0), (30.0, 48.0), (43.0, 67.0), (58.0, 48.0), (58.0, 27.0), (37.0, 69.0), (38.0, 46.0), (46.0, 10.0), (61.0, 33.0), (62.0, 63.0), (63.0, 69.0), (32.0, 22.0), (45.0, 35.0), (59.0, 15.0), (5.0, 6.0), (10.0, 17.0), (21.0, 10.0), (5.0, 64.0), (30.0, 15.0), (39.0, 10.0), (32.0, 39.0), (25.0, 32.0), (25.0, 55.0), (48.0, 28.0), (56.0, 37.0), (30.0, 40.0), (37.0, 52.0), (49.0, 49.0), (52.0, 64.0), (20.0, 26.0), (40.0, 30.0), (21.0, 47.0), (17.0, 63.0), (31.0, 62.0), (52.0, 33.0), (51.0, 21.0), (42.0, 41.0), (31.0, 32.0), (5.0, 25.0), (12.0, 42.0), (36.0, 16.0), (52.0, 41.0), (27.0, 23.0), (17.0, 33.0), (13.0, 13.0), (57.0, 58.0), (62.0, 42.0), (42.0, 57.0), (16.0, 57.0), (8.0, 52.0), (7.0, 38.0)], './data/tsp_200_2': [(2995.0, 264.0), (202.0, 233.0), (981.0, 848.0), (1346.0, 408.0), (781.0, 670.0), (1009.0, 1001.0), (2927.0, 1777.0), (2982.0, 949.0), (555.0, 1121.0), (464.0, 1302.0), (3452.0, 637.0), (571.0, 1982.0), (2656.0, 128.0), (1623.0, 1723.0), (2067.0, 694.0), (1725.0, 927.0), (3600.0, 459.0), (1109.0, 1196.0), (366.0, 339.0), (778.0, 1282.0), (386.0, 1616.0), (3918.0, 1217.0), (3332.0, 1049.0), (2597.0, 349.0), (811.0, 1295.0), (241.0, 1069.0), (2658.0, 360.0), (394.0, 1944.0), (3786.0, 1862.0), (264.0, 36.0), (2050.0, 1833.0), (3538.0, 125.0), (1646.0, 1817.0), (2993.0, 624.0), (547.0, 25.0), (3373.0, 1902.0), (460.0, 267.0), (3060.0, 781.0), (1828.0, 456.0), (1021.0, 962.0), (2347.0, 388.0), (3535.0, 1112.0), (1529.0, 581.0), (1203.0, 385.0), (1787.0, 1902.0), (2740.0, 1101.0), (555.0, 1753.0), (47.0, 363.0), (3935.0, 540.0), (3062.0, 329.0), (387.0, 199.0), (2901.0, 920.0), (931.0, 512.0), (1766.0, 692.0), (401.0, 980.0), (149.0, 1629.0), (2214.0, 1977.0), (3805.0, 1619.0), (1179.0, 969.0), (1017.0, 333.0), (2834.0, 1512.0), (634.0, 294.0), (1819.0, 814.0), (1393.0, 859.0), (1768.0, 1578.0), (3023.0, 871.0), (3248.0, 1906.0), (1632.0, 1742.0), (2223.0, 990.0), (3868.0, 697.0), (1541.0, 354.0), (2374.0, 1944.0), (1962.0, 389.0), (3007.0, 1524.0), (3220.0, 1945.0), (2356.0, 1568.0), (1604.0, 706.0), (2028.0, 1736.0), (2581.0, 121.0), (2221.0, 1578.0), (2944.0, 632.0), (1082.0, 1561.0), (997.0, 942.0), (2334.0, 523.0), (1264.0, 1090.0), (1699.0, 1294.0), (235.0, 1059.0), (2592.0, 248.0), (3642.0, 699.0), (3599.0, 514.0), (1766.0, 678.0), (240.0, 619.0), (1272.0, 246.0), (3503.0, 301.0), (80.0, 1533.0), (1677.0, 1238.0), (3766.0, 154.0), (3946.0, 459.0), (1994.0, 1852.0), (278.0, 165.0), (3140.0, 1401.0), (556.0, 1056.0), (3675.0, 1522.0), (1182.0, 1853.0), (3595.0, 111.0), (962.0, 1895.0), (2030.0, 1186.0), (3507.0, 1851.0), (2642.0, 1269.0), (3438.0, 901.0), (3858.0, 1472.0), (2937.0, 1568.0), (376.0, 1018.0), (839.0, 1355.0), (706.0, 1925.0), (749.0, 920.0), (298.0, 615.0), (694.0, 552.0), (387.0, 190.0), (2801.0, 695.0), (3133.0, 1143.0), (1517.0, 266.0), (1538.0, 224.0), (844.0, 520.0), (2639.0, 1239.0), (3123.0, 217.0), (2489.0, 1520.0), (3834.0, 1827.0), (3417.0, 1808.0), (2938.0, 543.0), (71.0, 1323.0), (3245.0, 1828.0), (731.0, 1741.0), (2312.0, 1270.0), (2426.0, 1851.0), (380.0, 478.0), (2310.0, 635.0), (2830.0, 775.0), (3829.0, 513.0), (3684.0, 445.0), (171.0, 514.0), (627.0, 1261.0), (1490.0, 1123.0), (61.0, 81.0), (422.0, 542.0), (2698.0, 1221.0), (2372.0, 127.0), (177.0, 1390.0), (3084.0, 748.0), (1213.0, 910.0), (3.0, 1817.0), (1782.0, 995.0), (3896.0, 742.0), (1829.0, 812.0), (1286.0, 550.0), (3017.0, 108.0), (2132.0, 1432.0), (2000.0, 1110.0), (3317.0, 1966.0), (1729.0, 1498.0), (2408.0, 1747.0), (3292.0, 152.0), (193.0, 1210.0), (782.0, 1462.0), (2503.0, 352.0), (1697.0, 1924.0), (3821.0, 147.0), (3370.0, 791.0), (3162.0, 367.0), (3938.0, 516.0), (2741.0, 1583.0), (2330.0, 741.0), (3918.0, 1088.0), (1794.0, 1589.0), (2929.0, 485.0), (3453.0, 1998.0), (896.0, 705.0), (399.0, 850.0), (2614.0, 195.0), (2800.0, 653.0), (2630.0, 20.0), (563.0, 1513.0), (1090.0, 1652.0), (2009.0, 1163.0), (3876.0, 1165.0), (3084.0, 774.0), (1526.0, 1612.0), (1612.0, 328.0), (1423.0, 1322.0), (3058.0, 1276.0), (3782.0, 1865.0), (347.0, 252.0), (3904.0, 1444.0), (2191.0, 1579.0), (3220.0, 1454.0), (468.0, 319.0), (3611.0, 1968.0), (3114.0, 1629.0), (3515.0, 1892.0), (3060.0, 155.0)], './data/tsp_100_3': [(86.0, 1065.0), (14.0, 454.0), (1327.0, 1893.0), (2773.0, 1286.0), (2469.0, 1838.0), (3835.0, 963.0), (1031.0, 428.0), (3853.0, 1712.0), (1868.0, 197.0), (1544.0, 863.0), (457.0, 1607.0), (3174.0, 1064.0), (192.0, 1004.0), (2318.0, 1925.0), (2232.0, 1374.0), (396.0, 828.0), (2365.0, 1649.0), (2499.0, 658.0), (1410.0, 307.0), (2990.0, 214.0), (3646.0, 1018.0), (3394.0, 1028.0), (1779.0, 90.0), (1058.0, 372.0), (2933.0, 1459.0), (3099.0, 173.0), (2178.0, 978.0), (138.0, 1610.0), (2082.0, 1753.0), (2302.0, 1127.0), (805.0, 272.0), (22.0, 1617.0), (3213.0, 1085.0), (99.0, 536.0), (1533.0, 1780.0), (3564.0, 676.0), (29.0, 6.0), (3808.0, 1375.0), (2221.0, 291.0), (3499.0, 1885.0), (3124.0, 408.0), (781.0, 671.0), (1027.0, 1041.0), (3249.0, 378.0), (3297.0, 491.0), (213.0, 220.0), (721.0, 186.0), (3736.0, 1542.0), (868.0, 731.0), (960.0, 303.0), (1357.0, 1905.0), (2650.0, 802.0), (1774.0, 107.0), (1307.0, 964.0), (3806.0, 746.0), (2687.0, 1353.0), (43.0, 1957.0), (3092.0, 1668.0), (185.0, 1542.0), (834.0, 629.0), (40.0, 462.0), (1183.0, 1391.0), (2048.0, 1628.0), (1097.0, 643.0), (1838.0, 1732.0), (234.0, 1118.0), (3314.0, 1881.0), (737.0, 1285.0), (779.0, 777.0), (2312.0, 1949.0), (2576.0, 189.0), (3078.0, 1541.0), (2781.0, 478.0), (705.0, 1812.0), (3409.0, 1917.0), (323.0, 1714.0), (1660.0, 1556.0), (3729.0, 1188.0), (693.0, 1383.0), (2361.0, 640.0), (2433.0, 1538.0), (554.0, 1825.0), (913.0, 317.0), (3586.0, 1909.0), (2636.0, 727.0), (1000.0, 457.0), (482.0, 1337.0), (3704.0, 1082.0), (3635.0, 1174.0), (1362.0, 1526.0), (2049.0, 417.0), (2552.0, 1909.0), (3939.0, 640.0), (219.0, 898.0), (812.0, 351.0), (901.0, 1552.0), (2513.0, 1572.0), (242.0, 584.0), (826.0, 1226.0), (3278.0, 799.0)]}
