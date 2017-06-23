
void Function_flux(int n, double* Tensor_8, double* Tensor_9, double* Tensor_10, double* Tensor_11, double* Tensor_12, double* Tensor_13, double* Tensor_14, double* Tensor_15, double* Tensor_16, double* Tensor_17, double* Tensor_18, double* Tensor_19, double* Tensor_0, double* Tensor_1, double* Tensor_2, double* Tensor_3, double* Tensor_4, double* Tensor_5, double* Tensor_6, double* Tensor_7, double* Tensor_394, double* Tensor_395, double* Tensor_400, double* Tensor_402, double* Tensor_406, double* Tensor_407) {
long long start = current_timestamp();for (int i = 0; i < n; i++) {
double Intermediate_0 = *(Tensor_0 + i*1 + 0);
double Intermediate_1 = *(Tensor_11 + i*1 + 0);
double Intermediate_2 = *(Tensor_10 + i*1 + 0);
double Intermediate_3 = *(Tensor_17 + i*3 + 2);
double Intermediate_4 = *(Tensor_7 + i*6 + 5);
double Intermediate_5 = Intermediate_4*Intermediate_3;
double Intermediate_6 = *(Tensor_17 + i*3 + 1);
double Intermediate_7 = *(Tensor_7 + i*6 + 4);
double Intermediate_8 = Intermediate_7*Intermediate_6;
double Intermediate_9 = *(Tensor_17 + i*3 + 0);
double Intermediate_10 = *(Tensor_7 + i*6 + 3);
double Intermediate_11 = Intermediate_10*Intermediate_9;
double Intermediate_12 = *(Tensor_16 + i*3 + 2);
double Intermediate_13 = *(Tensor_7 + i*6 + 2);
double Intermediate_14 = Intermediate_13*Intermediate_12;
double Intermediate_15 = *(Tensor_16 + i*3 + 1);
double Intermediate_16 = *(Tensor_7 + i*6 + 1);
double Intermediate_17 = Intermediate_16*Intermediate_15;
double Intermediate_18 = *(Tensor_16 + i*3 + 0);
double Intermediate_19 = *(Tensor_7 + i*6 + 0);
double Intermediate_20 = Intermediate_19*Intermediate_18;
double Intermediate_21 = *(Tensor_6 + i*2 + 1);
double Intermediate_22 = -1;
double Intermediate_23 = Intermediate_22*Intermediate_1;
double Intermediate_24 = Intermediate_23+Intermediate_2;
double Intermediate_25 = Intermediate_24*Intermediate_21;
double Intermediate_26 = *(Tensor_6 + i*2 + 0);
double Intermediate_27 = Intermediate_22*Intermediate_2;
double Intermediate_28 = Intermediate_27+Intermediate_1;
double Intermediate_29 = Intermediate_28*Intermediate_26;
double Intermediate_30 = 0.500025;
double Intermediate_31 = Intermediate_30+Intermediate_29+Intermediate_25+Intermediate_20+Intermediate_17+Intermediate_14+Intermediate_11+Intermediate_8+Intermediate_5+Intermediate_2+Intermediate_1;
double Intermediate_32 = *(Tensor_4 + i*1 + 0);
double Intermediate_33 = pow(Intermediate_32,Intermediate_22);
double Intermediate_34 = 0.5;
double Intermediate_35 = Intermediate_34+Intermediate_29+Intermediate_25+Intermediate_20+Intermediate_17+Intermediate_14+Intermediate_11+Intermediate_8+Intermediate_5+Intermediate_2+Intermediate_1;
double Intermediate_36 = pow(Intermediate_35,Intermediate_22);
double Intermediate_37 = -1435.0;
double Intermediate_38 = Intermediate_37*Intermediate_36*Intermediate_33*Intermediate_28*Intermediate_31;
double Intermediate_39 = *(Tensor_9 + i*3 + 2);
double Intermediate_40 = *(Tensor_8 + i*3 + 2);
double Intermediate_41 = *(Tensor_15 + i*9 + 8);
double Intermediate_42 = Intermediate_4*Intermediate_41;
double Intermediate_43 = *(Tensor_15 + i*9 + 7);
double Intermediate_44 = Intermediate_7*Intermediate_43;
double Intermediate_45 = *(Tensor_15 + i*9 + 6);
double Intermediate_46 = Intermediate_10*Intermediate_45;
double Intermediate_47 = *(Tensor_14 + i*9 + 8);
double Intermediate_48 = Intermediate_13*Intermediate_47;
double Intermediate_49 = *(Tensor_14 + i*9 + 7);
double Intermediate_50 = Intermediate_16*Intermediate_49;
double Intermediate_51 = *(Tensor_14 + i*9 + 6);
double Intermediate_52 = Intermediate_19*Intermediate_51;
double Intermediate_53 = Intermediate_22*Intermediate_39;
double Intermediate_54 = Intermediate_53+Intermediate_40;
double Intermediate_55 = Intermediate_54*Intermediate_21;
double Intermediate_56 = Intermediate_22*Intermediate_40;
double Intermediate_57 = Intermediate_56+Intermediate_39;
double Intermediate_58 = Intermediate_57*Intermediate_26;
double Intermediate_59 = Intermediate_34+Intermediate_58+Intermediate_55+Intermediate_52+Intermediate_50+Intermediate_48+Intermediate_46+Intermediate_44+Intermediate_42+Intermediate_40+Intermediate_39;
double Intermediate_60 = *(Tensor_5 + i*3 + 2);
double Intermediate_61 = *(Tensor_15 + i*9 + 4);
double Intermediate_62 = *(Tensor_15 + i*9 + 0);
double Intermediate_63 = *(Tensor_14 + i*9 + 4);
double Intermediate_64 = *(Tensor_14 + i*9 + 0);
double Intermediate_65 = 2.16666666666667;
double Intermediate_66 = Intermediate_65+Intermediate_64+Intermediate_63+Intermediate_47+Intermediate_62+Intermediate_61+Intermediate_41;
double Intermediate_67 = Intermediate_22*Intermediate_66*Intermediate_60;
double Intermediate_68 = *(Tensor_5 + i*3 + 1);
double Intermediate_69 = *(Tensor_15 + i*9 + 5);
double Intermediate_70 = *(Tensor_14 + i*9 + 5);
double Intermediate_71 = 1.0;
double Intermediate_72 = Intermediate_71+Intermediate_70+Intermediate_49+Intermediate_69+Intermediate_43;
double Intermediate_73 = Intermediate_72*Intermediate_68;
double Intermediate_74 = *(Tensor_5 + i*3 + 0);
double Intermediate_75 = *(Tensor_15 + i*9 + 2);
double Intermediate_76 = *(Tensor_14 + i*9 + 2);
double Intermediate_77 = Intermediate_71+Intermediate_76+Intermediate_51+Intermediate_75+Intermediate_45;
double Intermediate_78 = Intermediate_77*Intermediate_74;
double Intermediate_79 = 2;
double Intermediate_80 = Intermediate_79*Intermediate_41;
double Intermediate_81 = Intermediate_79*Intermediate_47;
double Intermediate_82 = Intermediate_71+Intermediate_81+Intermediate_80;
double Intermediate_83 = Intermediate_82*Intermediate_60;
double Intermediate_84 = Intermediate_83+Intermediate_78+Intermediate_73+Intermediate_67;
double Intermediate_85 = Intermediate_22*Intermediate_36*Intermediate_84*Intermediate_59*Intermediate_31;
double Intermediate_86 = *(Tensor_9 + i*3 + 1);
double Intermediate_87 = *(Tensor_8 + i*3 + 1);
double Intermediate_88 = Intermediate_4*Intermediate_69;
double Intermediate_89 = Intermediate_7*Intermediate_61;
double Intermediate_90 = *(Tensor_15 + i*9 + 3);
double Intermediate_91 = Intermediate_10*Intermediate_90;
double Intermediate_92 = Intermediate_13*Intermediate_70;
double Intermediate_93 = Intermediate_16*Intermediate_63;
double Intermediate_94 = *(Tensor_14 + i*9 + 3);
double Intermediate_95 = Intermediate_19*Intermediate_94;
double Intermediate_96 = Intermediate_22*Intermediate_86;
double Intermediate_97 = Intermediate_96+Intermediate_87;
double Intermediate_98 = Intermediate_97*Intermediate_21;
double Intermediate_99 = Intermediate_22*Intermediate_87;
double Intermediate_100 = Intermediate_99+Intermediate_86;
double Intermediate_101 = Intermediate_100*Intermediate_26;
double Intermediate_102 = Intermediate_34+Intermediate_101+Intermediate_98+Intermediate_95+Intermediate_93+Intermediate_92+Intermediate_91+Intermediate_89+Intermediate_88+Intermediate_87+Intermediate_86;
double Intermediate_103 = Intermediate_22*Intermediate_66*Intermediate_68;
double Intermediate_104 = Intermediate_72*Intermediate_60;
double Intermediate_105 = *(Tensor_15 + i*9 + 1);
double Intermediate_106 = *(Tensor_14 + i*9 + 1);
double Intermediate_107 = Intermediate_71+Intermediate_106+Intermediate_94+Intermediate_105+Intermediate_90;
double Intermediate_108 = Intermediate_107*Intermediate_74;
double Intermediate_109 = Intermediate_79*Intermediate_61;
double Intermediate_110 = Intermediate_79*Intermediate_63;
double Intermediate_111 = Intermediate_71+Intermediate_110+Intermediate_109;
double Intermediate_112 = Intermediate_111*Intermediate_68;
double Intermediate_113 = Intermediate_112+Intermediate_108+Intermediate_104+Intermediate_103;
double Intermediate_114 = Intermediate_22*Intermediate_36*Intermediate_113*Intermediate_102*Intermediate_31;
double Intermediate_115 = *(Tensor_9 + i*3 + 0);
double Intermediate_116 = *(Tensor_8 + i*3 + 0);
double Intermediate_117 = Intermediate_4*Intermediate_75;
double Intermediate_118 = Intermediate_7*Intermediate_105;
double Intermediate_119 = Intermediate_10*Intermediate_62;
double Intermediate_120 = Intermediate_13*Intermediate_76;
double Intermediate_121 = Intermediate_16*Intermediate_106;
double Intermediate_122 = Intermediate_19*Intermediate_64;
double Intermediate_123 = Intermediate_22*Intermediate_115;
double Intermediate_124 = Intermediate_123+Intermediate_116;
double Intermediate_125 = Intermediate_124*Intermediate_21;
double Intermediate_126 = Intermediate_22*Intermediate_116;
double Intermediate_127 = Intermediate_126+Intermediate_115;
double Intermediate_128 = Intermediate_127*Intermediate_26;
double Intermediate_129 = Intermediate_34+Intermediate_128+Intermediate_125+Intermediate_122+Intermediate_121+Intermediate_120+Intermediate_119+Intermediate_118+Intermediate_117+Intermediate_116+Intermediate_115;
double Intermediate_130 = Intermediate_22*Intermediate_66*Intermediate_74;
double Intermediate_131 = Intermediate_77*Intermediate_60;
double Intermediate_132 = Intermediate_107*Intermediate_68;
double Intermediate_133 = Intermediate_79*Intermediate_62;
double Intermediate_134 = Intermediate_79*Intermediate_64;
double Intermediate_135 = Intermediate_71+Intermediate_134+Intermediate_133;
double Intermediate_136 = Intermediate_135*Intermediate_74;
double Intermediate_137 = Intermediate_136+Intermediate_132+Intermediate_131+Intermediate_130;
double Intermediate_138 = Intermediate_22*Intermediate_36*Intermediate_137*Intermediate_129*Intermediate_31;
double Intermediate_139 = *(Tensor_13 + i*1 + 0);
double Intermediate_140 = *(Tensor_19 + i*3 + 2);
double Intermediate_141 = Intermediate_4*Intermediate_140;
double Intermediate_142 = *(Tensor_19 + i*3 + 1);
double Intermediate_143 = Intermediate_7*Intermediate_142;
double Intermediate_144 = *(Tensor_19 + i*3 + 0);
double Intermediate_145 = Intermediate_10*Intermediate_144;
double Intermediate_146 = *(Tensor_12 + i*1 + 0);
double Intermediate_147 = Intermediate_22*Intermediate_139;
double Intermediate_148 = Intermediate_147+Intermediate_146;
double Intermediate_149 = Intermediate_148*Intermediate_21;
double Intermediate_150 = Intermediate_149+Intermediate_145+Intermediate_143+Intermediate_141+Intermediate_139;
double Intermediate_151 = 1.4;
double Intermediate_152 = Intermediate_151+Intermediate_149+Intermediate_145+Intermediate_143+Intermediate_141+Intermediate_139;
double Intermediate_153 = 0.4;
double Intermediate_154 = Intermediate_153*Intermediate_4*Intermediate_3;
double Intermediate_155 = Intermediate_153*Intermediate_7*Intermediate_6;
double Intermediate_156 = Intermediate_153*Intermediate_10*Intermediate_9;
double Intermediate_157 = Intermediate_153*Intermediate_24*Intermediate_21;
double Intermediate_158 = Intermediate_153*Intermediate_1;
double Intermediate_159 = 287.0;
double Intermediate_160 = Intermediate_159+Intermediate_158+Intermediate_157+Intermediate_156+Intermediate_155+Intermediate_154;
double Intermediate_161 = pow(Intermediate_160,Intermediate_22);
double Intermediate_162 = Intermediate_161*Intermediate_150;
double Intermediate_163 = Intermediate_153+Intermediate_162;
double Intermediate_164 = pow(Intermediate_163,Intermediate_22);
double Intermediate_165 = Intermediate_164*Intermediate_152;
double Intermediate_166 = Intermediate_55+Intermediate_46+Intermediate_44+Intermediate_42+Intermediate_39;
double Intermediate_167 = pow(Intermediate_166,Intermediate_79);
double Intermediate_168 = Intermediate_98+Intermediate_91+Intermediate_89+Intermediate_88+Intermediate_86;
double Intermediate_169 = pow(Intermediate_168,Intermediate_79);
double Intermediate_170 = Intermediate_125+Intermediate_119+Intermediate_118+Intermediate_117+Intermediate_115;
double Intermediate_171 = pow(Intermediate_170,Intermediate_79);
double Intermediate_172 = Intermediate_34+Intermediate_171+Intermediate_169+Intermediate_167+Intermediate_165;
double Intermediate_173 = Intermediate_166*Intermediate_60;
double Intermediate_174 = Intermediate_168*Intermediate_68;
double Intermediate_175 = Intermediate_170*Intermediate_74;
double Intermediate_176 = Intermediate_175+Intermediate_174+Intermediate_173;
double Intermediate_177 = Intermediate_161*Intermediate_176*Intermediate_172*Intermediate_150;
double Intermediate_178 = *(Tensor_18 + i*3 + 2);
double Intermediate_179 = Intermediate_13*Intermediate_178;
double Intermediate_180 = *(Tensor_18 + i*3 + 1);
double Intermediate_181 = Intermediate_16*Intermediate_180;
double Intermediate_182 = *(Tensor_18 + i*3 + 0);
double Intermediate_183 = Intermediate_19*Intermediate_182;
double Intermediate_184 = Intermediate_22*Intermediate_146;
double Intermediate_185 = Intermediate_184+Intermediate_139;
double Intermediate_186 = Intermediate_185*Intermediate_26;
double Intermediate_187 = Intermediate_186+Intermediate_183+Intermediate_181+Intermediate_179+Intermediate_146;
double Intermediate_188 = Intermediate_151+Intermediate_186+Intermediate_183+Intermediate_181+Intermediate_179+Intermediate_146;
double Intermediate_189 = Intermediate_153*Intermediate_13*Intermediate_12;
double Intermediate_190 = Intermediate_153*Intermediate_16*Intermediate_15;
double Intermediate_191 = Intermediate_153*Intermediate_19*Intermediate_18;
double Intermediate_192 = Intermediate_153*Intermediate_28*Intermediate_26;
double Intermediate_193 = Intermediate_153*Intermediate_2;
double Intermediate_194 = Intermediate_159+Intermediate_193+Intermediate_192+Intermediate_191+Intermediate_190+Intermediate_189;
double Intermediate_195 = pow(Intermediate_194,Intermediate_22);
double Intermediate_196 = Intermediate_195*Intermediate_187;
double Intermediate_197 = Intermediate_153+Intermediate_196;
double Intermediate_198 = pow(Intermediate_197,Intermediate_22);
double Intermediate_199 = Intermediate_198*Intermediate_188;
double Intermediate_200 = Intermediate_58+Intermediate_52+Intermediate_50+Intermediate_48+Intermediate_40;
double Intermediate_201 = pow(Intermediate_200,Intermediate_79);
double Intermediate_202 = Intermediate_101+Intermediate_95+Intermediate_93+Intermediate_92+Intermediate_87;
double Intermediate_203 = pow(Intermediate_202,Intermediate_79);
double Intermediate_204 = Intermediate_128+Intermediate_122+Intermediate_121+Intermediate_120+Intermediate_116;
double Intermediate_205 = pow(Intermediate_204,Intermediate_79);
double Intermediate_206 = Intermediate_34+Intermediate_205+Intermediate_203+Intermediate_201+Intermediate_199;
double Intermediate_207 = Intermediate_200*Intermediate_60;
double Intermediate_208 = Intermediate_202*Intermediate_68;
double Intermediate_209 = Intermediate_204*Intermediate_74;
double Intermediate_210 = Intermediate_209+Intermediate_208+Intermediate_207;
double Intermediate_211 = Intermediate_195*Intermediate_210*Intermediate_206*Intermediate_187;
double Intermediate_212 = Intermediate_22*Intermediate_195*Intermediate_200*Intermediate_187;
double Intermediate_213 = Intermediate_161*Intermediate_166*Intermediate_150;
double Intermediate_214 = Intermediate_213+Intermediate_212;
double Intermediate_215 = Intermediate_22*Intermediate_214*Intermediate_60;
double Intermediate_216 = Intermediate_22*Intermediate_195*Intermediate_202*Intermediate_187;
double Intermediate_217 = Intermediate_161*Intermediate_168*Intermediate_150;
double Intermediate_218 = Intermediate_217+Intermediate_216;
double Intermediate_219 = Intermediate_22*Intermediate_218*Intermediate_68;
double Intermediate_220 = Intermediate_22*Intermediate_195*Intermediate_204*Intermediate_187;
double Intermediate_221 = Intermediate_161*Intermediate_170*Intermediate_150;
double Intermediate_222 = Intermediate_221+Intermediate_220;
double Intermediate_223 = Intermediate_22*Intermediate_222*Intermediate_74;
double Intermediate_224 = 0.5;
double Intermediate_225 = pow(Intermediate_162,Intermediate_224);
double Intermediate_226 = Intermediate_225*Intermediate_166;
double Intermediate_227 = pow(Intermediate_196,Intermediate_224);
double Intermediate_228 = Intermediate_227*Intermediate_200;
double Intermediate_229 = Intermediate_228+Intermediate_226;
double Intermediate_230 = Intermediate_227+Intermediate_225;
double Intermediate_231 = pow(Intermediate_230,Intermediate_22);
double Intermediate_232 = Intermediate_231*Intermediate_229*Intermediate_60;
double Intermediate_233 = Intermediate_225*Intermediate_168;
double Intermediate_234 = Intermediate_227*Intermediate_202;
double Intermediate_235 = Intermediate_234+Intermediate_233;
double Intermediate_236 = Intermediate_231*Intermediate_235*Intermediate_68;
double Intermediate_237 = Intermediate_225*Intermediate_170;
double Intermediate_238 = Intermediate_227*Intermediate_204;
double Intermediate_239 = Intermediate_238+Intermediate_237;
double Intermediate_240 = Intermediate_231*Intermediate_239*Intermediate_74;
double Intermediate_241 = Intermediate_240+Intermediate_236+Intermediate_232;
double Intermediate_242 = Intermediate_22*Intermediate_195*Intermediate_187;
double Intermediate_243 = Intermediate_162+Intermediate_242;
double Intermediate_244 = Intermediate_243*Intermediate_241;
double Intermediate_245 = Intermediate_244+Intermediate_223+Intermediate_219+Intermediate_215;

double Intermediate_247 = pow(Intermediate_229,Intermediate_79);
double Intermediate_248 = -2;
double Intermediate_249 = pow(Intermediate_230,Intermediate_248);
double Intermediate_250 = Intermediate_22*Intermediate_249*Intermediate_247;
double Intermediate_251 = pow(Intermediate_235,Intermediate_79);
double Intermediate_252 = Intermediate_22*Intermediate_249*Intermediate_251;
double Intermediate_253 = pow(Intermediate_239,Intermediate_79);
double Intermediate_254 = Intermediate_22*Intermediate_249*Intermediate_253;
double Intermediate_255 = Intermediate_225*Intermediate_172;
double Intermediate_256 = Intermediate_227*Intermediate_206;
double Intermediate_257 = Intermediate_256+Intermediate_255;
double Intermediate_258 = Intermediate_231*Intermediate_257;
double Intermediate_259 = -0.1;
double Intermediate_260 = Intermediate_259+Intermediate_258+Intermediate_254+Intermediate_252+Intermediate_250;
double Intermediate_261 = pow(Intermediate_260,Intermediate_224);
double Intermediate_262 = Intermediate_261+Intermediate_240+Intermediate_236+Intermediate_232;

double Intermediate_264 = 0;
int Intermediate_265 = Intermediate_262 < Intermediate_264;
double Intermediate_266 = Intermediate_22*Intermediate_231*Intermediate_229*Intermediate_60;
double Intermediate_267 = Intermediate_22*Intermediate_231*Intermediate_235*Intermediate_68;
double Intermediate_268 = Intermediate_22*Intermediate_231*Intermediate_239*Intermediate_74;
double Intermediate_269 = Intermediate_22*Intermediate_261;
double Intermediate_270 = Intermediate_269+Intermediate_268+Intermediate_267+Intermediate_266;


                double Intermediate_272;
                if (Intermediate_265) 
                    Intermediate_272 = Intermediate_270;
                else 
                    Intermediate_272 = Intermediate_262;
                

double Intermediate_274 = Intermediate_22*Intermediate_166*Intermediate_60;
double Intermediate_275 = Intermediate_22*Intermediate_168*Intermediate_68;
double Intermediate_276 = Intermediate_22*Intermediate_170*Intermediate_74;
double Intermediate_277 = Intermediate_209+Intermediate_208+Intermediate_207+Intermediate_276+Intermediate_275+Intermediate_274;

int Intermediate_279 = Intermediate_277 < Intermediate_264;
double Intermediate_280 = Intermediate_22*Intermediate_200*Intermediate_60;
double Intermediate_281 = Intermediate_22*Intermediate_202*Intermediate_68;
double Intermediate_282 = Intermediate_22*Intermediate_204*Intermediate_74;
double Intermediate_283 = Intermediate_175+Intermediate_174+Intermediate_173+Intermediate_282+Intermediate_281+Intermediate_280;


                double Intermediate_285;
                if (Intermediate_279) 
                    Intermediate_285 = Intermediate_283;
                else 
                    Intermediate_285 = Intermediate_277;
                
double Intermediate_286 = 2.0;
double Intermediate_287 = Intermediate_286*Intermediate_285;
double Intermediate_288 = pow(Intermediate_150,Intermediate_22);
double Intermediate_289 = Intermediate_288*Intermediate_152*Intermediate_160;
double Intermediate_290 = pow(Intermediate_289,Intermediate_224);
double Intermediate_291 = Intermediate_22*Intermediate_290;
double Intermediate_292 = pow(Intermediate_187,Intermediate_22);
double Intermediate_293 = Intermediate_292*Intermediate_188*Intermediate_194;
double Intermediate_294 = pow(Intermediate_293,Intermediate_224);
double Intermediate_295 = Intermediate_294+Intermediate_291;

int Intermediate_297 = Intermediate_295 < Intermediate_264;
double Intermediate_298 = Intermediate_22*Intermediate_294;
double Intermediate_299 = Intermediate_290+Intermediate_298;


                double Intermediate_301;
                if (Intermediate_297) 
                    Intermediate_301 = Intermediate_299;
                else 
                    Intermediate_301 = Intermediate_295;
                
double Intermediate_302 = Intermediate_286*Intermediate_301;
double Intermediate_303 = Intermediate_286+Intermediate_302+Intermediate_287;
int Intermediate_304 = Intermediate_272 < Intermediate_303;
double Intermediate_305 = 0.25;
double Intermediate_306 = Intermediate_305+Intermediate_272;
double Intermediate_307 = Intermediate_71+Intermediate_301+Intermediate_285;
double Intermediate_308 = pow(Intermediate_307,Intermediate_22);
double Intermediate_309 = Intermediate_308*Intermediate_306*Intermediate_272;
double Intermediate_310 = Intermediate_71+Intermediate_309+Intermediate_301+Intermediate_285;


                double Intermediate_312;
                if (Intermediate_304) 
                    Intermediate_312 = Intermediate_310;
                else 
                    Intermediate_312 = Intermediate_272;
                
double Intermediate_313 = Intermediate_269+Intermediate_240+Intermediate_236+Intermediate_232;

int Intermediate_315 = Intermediate_313 < Intermediate_264;
double Intermediate_316 = Intermediate_261+Intermediate_268+Intermediate_267+Intermediate_266;


                double Intermediate_318;
                if (Intermediate_315) 
                    Intermediate_318 = Intermediate_316;
                else 
                    Intermediate_318 = Intermediate_313;
                

int Intermediate_320 = Intermediate_318 < Intermediate_303;
double Intermediate_321 = Intermediate_305+Intermediate_318;
double Intermediate_322 = Intermediate_308*Intermediate_321*Intermediate_318;
double Intermediate_323 = Intermediate_71+Intermediate_322+Intermediate_301+Intermediate_285;


                double Intermediate_325;
                if (Intermediate_320) 
                    Intermediate_325 = Intermediate_323;
                else 
                    Intermediate_325 = Intermediate_318;
                
double Intermediate_326 = Intermediate_22*Intermediate_325;
double Intermediate_327 = Intermediate_34+Intermediate_326+Intermediate_312;
double Intermediate_328 = -0.5;
double Intermediate_329 = pow(Intermediate_260,Intermediate_328);
double Intermediate_330 = Intermediate_22*Intermediate_329*Intermediate_327*Intermediate_245;
double Intermediate_331 = Intermediate_22*Intermediate_195*Intermediate_206*Intermediate_187;
double Intermediate_332 = Intermediate_22*Intermediate_231*Intermediate_229*Intermediate_214;
double Intermediate_333 = Intermediate_22*Intermediate_231*Intermediate_235*Intermediate_218;
double Intermediate_334 = Intermediate_22*Intermediate_231*Intermediate_239*Intermediate_222;
double Intermediate_335 = Intermediate_161*Intermediate_172*Intermediate_150;
double Intermediate_336 = Intermediate_22*Intermediate_4*Intermediate_140;
double Intermediate_337 = Intermediate_22*Intermediate_7*Intermediate_142;
double Intermediate_338 = Intermediate_22*Intermediate_10*Intermediate_144;
double Intermediate_339 = Intermediate_22*Intermediate_148*Intermediate_21;
double Intermediate_340 = Intermediate_249*Intermediate_247;
double Intermediate_341 = Intermediate_249*Intermediate_251;
double Intermediate_342 = Intermediate_249*Intermediate_253;
double Intermediate_343 = Intermediate_34+Intermediate_342+Intermediate_341+Intermediate_340;
double Intermediate_344 = Intermediate_243*Intermediate_343;
double Intermediate_345 = Intermediate_153+Intermediate_147+Intermediate_186+Intermediate_344+Intermediate_183+Intermediate_181+Intermediate_179+Intermediate_339+Intermediate_338+Intermediate_337+Intermediate_336+Intermediate_335+Intermediate_334+Intermediate_333+Intermediate_332+Intermediate_331+Intermediate_146;

int Intermediate_347 = Intermediate_241 < Intermediate_264;
double Intermediate_348 = Intermediate_268+Intermediate_267+Intermediate_266;


                double Intermediate_350;
                if (Intermediate_347) 
                    Intermediate_350 = Intermediate_348;
                else 
                    Intermediate_350 = Intermediate_241;
                

int Intermediate_352 = Intermediate_350 < Intermediate_303;
double Intermediate_353 = Intermediate_305+Intermediate_350;
double Intermediate_354 = Intermediate_308*Intermediate_353*Intermediate_350;
double Intermediate_355 = Intermediate_71+Intermediate_354+Intermediate_301+Intermediate_285;


                double Intermediate_357;
                if (Intermediate_352) 
                    Intermediate_357 = Intermediate_355;
                else 
                    Intermediate_357 = Intermediate_350;
                
double Intermediate_358 = Intermediate_22*Intermediate_357;
double Intermediate_359 = Intermediate_34+Intermediate_358+Intermediate_325+Intermediate_312;
double Intermediate_360 = pow(Intermediate_260,Intermediate_22);
double Intermediate_361 = Intermediate_360*Intermediate_359*Intermediate_345;
double Intermediate_362 = Intermediate_361+Intermediate_330;
double Intermediate_363 = Intermediate_22*Intermediate_231*Intermediate_257*Intermediate_362;
double Intermediate_364 = Intermediate_147+Intermediate_186+Intermediate_183+Intermediate_181+Intermediate_179+Intermediate_339+Intermediate_338+Intermediate_337+Intermediate_336+Intermediate_335+Intermediate_331+Intermediate_146;
double Intermediate_365 = Intermediate_22*Intermediate_364*Intermediate_357;
double Intermediate_366 = Intermediate_22*Intermediate_329*Intermediate_327*Intermediate_345;
double Intermediate_367 = Intermediate_359*Intermediate_245;
double Intermediate_368 = Intermediate_367+Intermediate_366;
double Intermediate_369 = Intermediate_368*Intermediate_241;
double Intermediate_370 = Intermediate_369+Intermediate_365+Intermediate_363+Intermediate_211+Intermediate_177+Intermediate_138+Intermediate_114+Intermediate_85+Intermediate_38;
double Intermediate_371 = *(Tensor_2 + i*1 + 0);
double Intermediate_372 = pow(Intermediate_371,Intermediate_22);
double Intermediate_373 = Intermediate_22*Intermediate_372*Intermediate_370*Intermediate_0; *(Tensor_407 + i*1 + 0) = Intermediate_373;
double Intermediate_374 = *(Tensor_1 + i*1 + 0);
double Intermediate_375 = pow(Intermediate_374,Intermediate_22);
double Intermediate_376 = Intermediate_375*Intermediate_370*Intermediate_0; *(Tensor_406 + i*1 + 0) = Intermediate_376;
double Intermediate_377 = Intermediate_161*Intermediate_176*Intermediate_166*Intermediate_150;
double Intermediate_378 = Intermediate_195*Intermediate_210*Intermediate_200*Intermediate_187;
double Intermediate_379 = Intermediate_22*Intermediate_36*Intermediate_84*Intermediate_31;
double Intermediate_380 = Intermediate_22*Intermediate_231*Intermediate_229*Intermediate_362;
double Intermediate_381 = Intermediate_22*Intermediate_214*Intermediate_357;
double Intermediate_382 = Intermediate_186+Intermediate_149+Intermediate_183+Intermediate_181+Intermediate_179+Intermediate_145+Intermediate_143+Intermediate_141+Intermediate_146+Intermediate_139;
double Intermediate_383 = Intermediate_382*Intermediate_60;
double Intermediate_384 = Intermediate_368*Intermediate_60;
double Intermediate_385 = Intermediate_384+Intermediate_383+Intermediate_381+Intermediate_380+Intermediate_379+Intermediate_378+Intermediate_377;
double Intermediate_386 = Intermediate_22*Intermediate_372*Intermediate_385*Intermediate_0; *(Tensor_402 + i*3 + 2) = Intermediate_386;
double Intermediate_387 = Intermediate_161*Intermediate_176*Intermediate_168*Intermediate_150;
double Intermediate_388 = Intermediate_195*Intermediate_210*Intermediate_202*Intermediate_187;
double Intermediate_389 = Intermediate_22*Intermediate_36*Intermediate_113*Intermediate_31;
double Intermediate_390 = Intermediate_22*Intermediate_231*Intermediate_235*Intermediate_362;
double Intermediate_391 = Intermediate_22*Intermediate_218*Intermediate_357;
double Intermediate_392 = Intermediate_382*Intermediate_68;
double Intermediate_393 = Intermediate_368*Intermediate_68;
double Intermediate_394 = Intermediate_393+Intermediate_392+Intermediate_391+Intermediate_390+Intermediate_389+Intermediate_388+Intermediate_387;
double Intermediate_395 = Intermediate_22*Intermediate_372*Intermediate_394*Intermediate_0; *(Tensor_402 + i*3 + 1) = Intermediate_395;
double Intermediate_396 = Intermediate_161*Intermediate_176*Intermediate_170*Intermediate_150;
double Intermediate_397 = Intermediate_195*Intermediate_210*Intermediate_204*Intermediate_187;
double Intermediate_398 = Intermediate_22*Intermediate_36*Intermediate_137*Intermediate_31;
double Intermediate_399 = Intermediate_22*Intermediate_231*Intermediate_239*Intermediate_362;
double Intermediate_400 = Intermediate_22*Intermediate_222*Intermediate_357;
double Intermediate_401 = Intermediate_382*Intermediate_74;
double Intermediate_402 = Intermediate_368*Intermediate_74;
double Intermediate_403 = Intermediate_402+Intermediate_401+Intermediate_400+Intermediate_399+Intermediate_398+Intermediate_397+Intermediate_396;
double Intermediate_404 = Intermediate_22*Intermediate_372*Intermediate_403*Intermediate_0; *(Tensor_402 + i*3 + 0) = Intermediate_404;
double Intermediate_405 = Intermediate_375*Intermediate_385*Intermediate_0; *(Tensor_400 + i*3 + 2) = Intermediate_405;
double Intermediate_406 = Intermediate_375*Intermediate_394*Intermediate_0; *(Tensor_400 + i*3 + 1) = Intermediate_406;
double Intermediate_407 = Intermediate_375*Intermediate_403*Intermediate_0; *(Tensor_400 + i*3 + 0) = Intermediate_407;
double Intermediate_408 = Intermediate_22*Intermediate_360*Intermediate_359*Intermediate_345;
double Intermediate_409 = Intermediate_161*Intermediate_176*Intermediate_150;
double Intermediate_410 = Intermediate_195*Intermediate_210*Intermediate_187;
double Intermediate_411 = Intermediate_329*Intermediate_327*Intermediate_245;
double Intermediate_412 = Intermediate_22*Intermediate_243*Intermediate_357;
double Intermediate_413 = Intermediate_412+Intermediate_411+Intermediate_410+Intermediate_409+Intermediate_408;
double Intermediate_414 = Intermediate_22*Intermediate_372*Intermediate_413*Intermediate_0; *(Tensor_395 + i*1 + 0) = Intermediate_414;
double Intermediate_415 = Intermediate_375*Intermediate_413*Intermediate_0; *(Tensor_394 + i*1 + 0) = Intermediate_415;
}
long long end = current_timestamp(); mil += end-start; printf("c module: %lld\n", mil);}
void Function_characteristicFlux(int n, double* Tensor_408, double* Tensor_409, double* Tensor_410, double* Tensor_411, double* Tensor_412, double* Tensor_413, double* Tensor_414, double* Tensor_415, double* Tensor_416, double* Tensor_417, double* Tensor_418, double* Tensor_419, double* Tensor_0, double* Tensor_1, double* Tensor_2, double* Tensor_3, double* Tensor_4, double* Tensor_5, double* Tensor_6, double* Tensor_7, double* Tensor_793, double* Tensor_794, double* Tensor_798, double* Tensor_800, double* Tensor_803, double* Tensor_804) {
long long start = current_timestamp();for (int i = 0; i < n; i++) {
double Intermediate_0 = 0; *(Tensor_804 + i*1 + 0) = Intermediate_0;
double Intermediate_1 = *(Tensor_0 + i*1 + 0);
double Intermediate_2 = *(Tensor_411 + i*1 + 0);
double Intermediate_3 = *(Tensor_410 + i*1 + 0);
double Intermediate_4 = *(Tensor_417 + i*3 + 2);
double Intermediate_5 = *(Tensor_7 + i*6 + 5);
double Intermediate_6 = Intermediate_5*Intermediate_4;
double Intermediate_7 = *(Tensor_417 + i*3 + 1);
double Intermediate_8 = *(Tensor_7 + i*6 + 4);
double Intermediate_9 = Intermediate_8*Intermediate_7;
double Intermediate_10 = *(Tensor_417 + i*3 + 0);
double Intermediate_11 = *(Tensor_7 + i*6 + 3);
double Intermediate_12 = Intermediate_11*Intermediate_10;
double Intermediate_13 = *(Tensor_416 + i*3 + 2);
double Intermediate_14 = *(Tensor_7 + i*6 + 2);
double Intermediate_15 = Intermediate_14*Intermediate_13;
double Intermediate_16 = *(Tensor_416 + i*3 + 1);
double Intermediate_17 = *(Tensor_7 + i*6 + 1);
double Intermediate_18 = Intermediate_17*Intermediate_16;
double Intermediate_19 = *(Tensor_416 + i*3 + 0);
double Intermediate_20 = *(Tensor_7 + i*6 + 0);
double Intermediate_21 = Intermediate_20*Intermediate_19;
double Intermediate_22 = *(Tensor_6 + i*2 + 1);
double Intermediate_23 = -1;
double Intermediate_24 = Intermediate_23*Intermediate_2;
double Intermediate_25 = Intermediate_24+Intermediate_3;
double Intermediate_26 = Intermediate_25*Intermediate_22;
double Intermediate_27 = *(Tensor_6 + i*2 + 0);
double Intermediate_28 = Intermediate_23*Intermediate_3;
double Intermediate_29 = Intermediate_28+Intermediate_2;
double Intermediate_30 = Intermediate_29*Intermediate_27;
double Intermediate_31 = 0.500025;
double Intermediate_32 = Intermediate_31+Intermediate_30+Intermediate_26+Intermediate_21+Intermediate_18+Intermediate_15+Intermediate_12+Intermediate_9+Intermediate_6+Intermediate_3+Intermediate_2;
double Intermediate_33 = *(Tensor_4 + i*1 + 0);
double Intermediate_34 = pow(Intermediate_33,Intermediate_23);
double Intermediate_35 = 0.5;
double Intermediate_36 = Intermediate_35+Intermediate_30+Intermediate_26+Intermediate_21+Intermediate_18+Intermediate_15+Intermediate_12+Intermediate_9+Intermediate_6+Intermediate_3+Intermediate_2;
double Intermediate_37 = pow(Intermediate_36,Intermediate_23);
double Intermediate_38 = -1435.0;
double Intermediate_39 = Intermediate_38*Intermediate_37*Intermediate_34*Intermediate_29*Intermediate_32;
double Intermediate_40 = *(Tensor_409 + i*3 + 2);
double Intermediate_41 = *(Tensor_408 + i*3 + 2);
double Intermediate_42 = *(Tensor_415 + i*9 + 8);
double Intermediate_43 = Intermediate_5*Intermediate_42;
double Intermediate_44 = *(Tensor_415 + i*9 + 7);
double Intermediate_45 = Intermediate_8*Intermediate_44;
double Intermediate_46 = *(Tensor_415 + i*9 + 6);
double Intermediate_47 = Intermediate_11*Intermediate_46;
double Intermediate_48 = *(Tensor_414 + i*9 + 8);
double Intermediate_49 = Intermediate_14*Intermediate_48;
double Intermediate_50 = *(Tensor_414 + i*9 + 7);
double Intermediate_51 = Intermediate_17*Intermediate_50;
double Intermediate_52 = *(Tensor_414 + i*9 + 6);
double Intermediate_53 = Intermediate_20*Intermediate_52;
double Intermediate_54 = Intermediate_23*Intermediate_40;
double Intermediate_55 = Intermediate_54+Intermediate_41;
double Intermediate_56 = Intermediate_55*Intermediate_22;
double Intermediate_57 = Intermediate_23*Intermediate_41;
double Intermediate_58 = Intermediate_57+Intermediate_40;
double Intermediate_59 = Intermediate_58*Intermediate_27;
double Intermediate_60 = Intermediate_35+Intermediate_59+Intermediate_56+Intermediate_53+Intermediate_51+Intermediate_49+Intermediate_47+Intermediate_45+Intermediate_43+Intermediate_41+Intermediate_40;
double Intermediate_61 = *(Tensor_5 + i*3 + 2);
double Intermediate_62 = *(Tensor_415 + i*9 + 4);
double Intermediate_63 = *(Tensor_415 + i*9 + 0);
double Intermediate_64 = *(Tensor_414 + i*9 + 4);
double Intermediate_65 = *(Tensor_414 + i*9 + 0);
double Intermediate_66 = 2.16666666666667;
double Intermediate_67 = Intermediate_66+Intermediate_65+Intermediate_64+Intermediate_48+Intermediate_63+Intermediate_62+Intermediate_42;
double Intermediate_68 = Intermediate_23*Intermediate_67*Intermediate_61;
double Intermediate_69 = *(Tensor_5 + i*3 + 1);
double Intermediate_70 = *(Tensor_415 + i*9 + 5);
double Intermediate_71 = *(Tensor_414 + i*9 + 5);
double Intermediate_72 = 1.0;
double Intermediate_73 = Intermediate_72+Intermediate_71+Intermediate_50+Intermediate_70+Intermediate_44;
double Intermediate_74 = Intermediate_73*Intermediate_69;
double Intermediate_75 = *(Tensor_5 + i*3 + 0);
double Intermediate_76 = *(Tensor_415 + i*9 + 2);
double Intermediate_77 = *(Tensor_414 + i*9 + 2);
double Intermediate_78 = Intermediate_72+Intermediate_77+Intermediate_52+Intermediate_76+Intermediate_46;
double Intermediate_79 = Intermediate_78*Intermediate_75;
double Intermediate_80 = 2;
double Intermediate_81 = Intermediate_80*Intermediate_42;
double Intermediate_82 = Intermediate_80*Intermediate_48;
double Intermediate_83 = Intermediate_72+Intermediate_82+Intermediate_81;
double Intermediate_84 = Intermediate_83*Intermediate_61;
double Intermediate_85 = Intermediate_84+Intermediate_79+Intermediate_74+Intermediate_68;
double Intermediate_86 = Intermediate_23*Intermediate_37*Intermediate_85*Intermediate_60*Intermediate_32;
double Intermediate_87 = *(Tensor_409 + i*3 + 1);
double Intermediate_88 = *(Tensor_408 + i*3 + 1);
double Intermediate_89 = Intermediate_5*Intermediate_70;
double Intermediate_90 = Intermediate_8*Intermediate_62;
double Intermediate_91 = *(Tensor_415 + i*9 + 3);
double Intermediate_92 = Intermediate_11*Intermediate_91;
double Intermediate_93 = Intermediate_14*Intermediate_71;
double Intermediate_94 = Intermediate_17*Intermediate_64;
double Intermediate_95 = *(Tensor_414 + i*9 + 3);
double Intermediate_96 = Intermediate_20*Intermediate_95;
double Intermediate_97 = Intermediate_23*Intermediate_87;
double Intermediate_98 = Intermediate_97+Intermediate_88;
double Intermediate_99 = Intermediate_98*Intermediate_22;
double Intermediate_100 = Intermediate_23*Intermediate_88;
double Intermediate_101 = Intermediate_100+Intermediate_87;
double Intermediate_102 = Intermediate_101*Intermediate_27;
double Intermediate_103 = Intermediate_35+Intermediate_102+Intermediate_99+Intermediate_96+Intermediate_94+Intermediate_93+Intermediate_92+Intermediate_90+Intermediate_89+Intermediate_88+Intermediate_87;
double Intermediate_104 = Intermediate_23*Intermediate_67*Intermediate_69;
double Intermediate_105 = Intermediate_73*Intermediate_61;
double Intermediate_106 = *(Tensor_415 + i*9 + 1);
double Intermediate_107 = *(Tensor_414 + i*9 + 1);
double Intermediate_108 = Intermediate_72+Intermediate_107+Intermediate_95+Intermediate_106+Intermediate_91;
double Intermediate_109 = Intermediate_108*Intermediate_75;
double Intermediate_110 = Intermediate_80*Intermediate_62;
double Intermediate_111 = Intermediate_80*Intermediate_64;
double Intermediate_112 = Intermediate_72+Intermediate_111+Intermediate_110;
double Intermediate_113 = Intermediate_112*Intermediate_69;
double Intermediate_114 = Intermediate_113+Intermediate_109+Intermediate_105+Intermediate_104;
double Intermediate_115 = Intermediate_23*Intermediate_37*Intermediate_114*Intermediate_103*Intermediate_32;
double Intermediate_116 = *(Tensor_409 + i*3 + 0);
double Intermediate_117 = *(Tensor_408 + i*3 + 0);
double Intermediate_118 = Intermediate_5*Intermediate_76;
double Intermediate_119 = Intermediate_8*Intermediate_106;
double Intermediate_120 = Intermediate_11*Intermediate_63;
double Intermediate_121 = Intermediate_14*Intermediate_77;
double Intermediate_122 = Intermediate_17*Intermediate_107;
double Intermediate_123 = Intermediate_20*Intermediate_65;
double Intermediate_124 = Intermediate_23*Intermediate_116;
double Intermediate_125 = Intermediate_124+Intermediate_117;
double Intermediate_126 = Intermediate_125*Intermediate_22;
double Intermediate_127 = Intermediate_23*Intermediate_117;
double Intermediate_128 = Intermediate_127+Intermediate_116;
double Intermediate_129 = Intermediate_128*Intermediate_27;
double Intermediate_130 = Intermediate_35+Intermediate_129+Intermediate_126+Intermediate_123+Intermediate_122+Intermediate_121+Intermediate_120+Intermediate_119+Intermediate_118+Intermediate_117+Intermediate_116;
double Intermediate_131 = Intermediate_23*Intermediate_67*Intermediate_75;
double Intermediate_132 = Intermediate_78*Intermediate_61;
double Intermediate_133 = Intermediate_108*Intermediate_69;
double Intermediate_134 = Intermediate_80*Intermediate_63;
double Intermediate_135 = Intermediate_80*Intermediate_65;
double Intermediate_136 = Intermediate_72+Intermediate_135+Intermediate_134;
double Intermediate_137 = Intermediate_136*Intermediate_75;
double Intermediate_138 = Intermediate_137+Intermediate_133+Intermediate_132+Intermediate_131;
double Intermediate_139 = Intermediate_23*Intermediate_37*Intermediate_138*Intermediate_130*Intermediate_32;
double Intermediate_140 = *(Tensor_413 + i*1 + 0);
double Intermediate_141 = *(Tensor_419 + i*3 + 2);
double Intermediate_142 = Intermediate_5*Intermediate_141;
double Intermediate_143 = *(Tensor_419 + i*3 + 1);
double Intermediate_144 = Intermediate_8*Intermediate_143;
double Intermediate_145 = *(Tensor_419 + i*3 + 0);
double Intermediate_146 = Intermediate_11*Intermediate_145;
double Intermediate_147 = *(Tensor_412 + i*1 + 0);
double Intermediate_148 = Intermediate_23*Intermediate_140;
double Intermediate_149 = Intermediate_148+Intermediate_147;
double Intermediate_150 = Intermediate_149*Intermediate_22;
double Intermediate_151 = Intermediate_150+Intermediate_146+Intermediate_144+Intermediate_142+Intermediate_140;
double Intermediate_152 = 1.4;
double Intermediate_153 = Intermediate_152+Intermediate_150+Intermediate_146+Intermediate_144+Intermediate_142+Intermediate_140;
double Intermediate_154 = 0.4;
double Intermediate_155 = Intermediate_154*Intermediate_5*Intermediate_4;
double Intermediate_156 = Intermediate_154*Intermediate_8*Intermediate_7;
double Intermediate_157 = Intermediate_154*Intermediate_11*Intermediate_10;
double Intermediate_158 = Intermediate_154*Intermediate_25*Intermediate_22;
double Intermediate_159 = Intermediate_154*Intermediate_2;
double Intermediate_160 = 287.0;
double Intermediate_161 = Intermediate_160+Intermediate_159+Intermediate_158+Intermediate_157+Intermediate_156+Intermediate_155;
double Intermediate_162 = pow(Intermediate_161,Intermediate_23);
double Intermediate_163 = Intermediate_162*Intermediate_151;
double Intermediate_164 = Intermediate_154+Intermediate_163;
double Intermediate_165 = pow(Intermediate_164,Intermediate_23);
double Intermediate_166 = Intermediate_165*Intermediate_153;
double Intermediate_167 = Intermediate_56+Intermediate_47+Intermediate_45+Intermediate_43+Intermediate_40;
double Intermediate_168 = pow(Intermediate_167,Intermediate_80);
double Intermediate_169 = Intermediate_99+Intermediate_92+Intermediate_90+Intermediate_89+Intermediate_87;
double Intermediate_170 = pow(Intermediate_169,Intermediate_80);
double Intermediate_171 = Intermediate_126+Intermediate_120+Intermediate_119+Intermediate_118+Intermediate_116;
double Intermediate_172 = pow(Intermediate_171,Intermediate_80);
double Intermediate_173 = Intermediate_35+Intermediate_172+Intermediate_170+Intermediate_168+Intermediate_166;
double Intermediate_174 = Intermediate_167*Intermediate_61;
double Intermediate_175 = Intermediate_169*Intermediate_69;
double Intermediate_176 = Intermediate_171*Intermediate_75;
double Intermediate_177 = Intermediate_176+Intermediate_175+Intermediate_174;
double Intermediate_178 = Intermediate_162*Intermediate_177*Intermediate_173*Intermediate_151;
double Intermediate_179 = *(Tensor_418 + i*3 + 2);
double Intermediate_180 = Intermediate_14*Intermediate_179;
double Intermediate_181 = *(Tensor_418 + i*3 + 1);
double Intermediate_182 = Intermediate_17*Intermediate_181;
double Intermediate_183 = *(Tensor_418 + i*3 + 0);
double Intermediate_184 = Intermediate_20*Intermediate_183;
double Intermediate_185 = Intermediate_23*Intermediate_147;
double Intermediate_186 = Intermediate_185+Intermediate_140;
double Intermediate_187 = Intermediate_186*Intermediate_27;
double Intermediate_188 = Intermediate_187+Intermediate_184+Intermediate_182+Intermediate_180+Intermediate_147;
double Intermediate_189 = Intermediate_152+Intermediate_187+Intermediate_184+Intermediate_182+Intermediate_180+Intermediate_147;
double Intermediate_190 = Intermediate_154*Intermediate_14*Intermediate_13;
double Intermediate_191 = Intermediate_154*Intermediate_17*Intermediate_16;
double Intermediate_192 = Intermediate_154*Intermediate_20*Intermediate_19;
double Intermediate_193 = Intermediate_154*Intermediate_29*Intermediate_27;
double Intermediate_194 = Intermediate_154*Intermediate_3;
double Intermediate_195 = Intermediate_160+Intermediate_194+Intermediate_193+Intermediate_192+Intermediate_191+Intermediate_190;
double Intermediate_196 = pow(Intermediate_195,Intermediate_23);
double Intermediate_197 = Intermediate_196*Intermediate_188;
double Intermediate_198 = Intermediate_154+Intermediate_197;
double Intermediate_199 = pow(Intermediate_198,Intermediate_23);
double Intermediate_200 = Intermediate_199*Intermediate_189;
double Intermediate_201 = Intermediate_59+Intermediate_53+Intermediate_51+Intermediate_49+Intermediate_41;
double Intermediate_202 = pow(Intermediate_201,Intermediate_80);
double Intermediate_203 = Intermediate_102+Intermediate_96+Intermediate_94+Intermediate_93+Intermediate_88;
double Intermediate_204 = pow(Intermediate_203,Intermediate_80);
double Intermediate_205 = Intermediate_129+Intermediate_123+Intermediate_122+Intermediate_121+Intermediate_117;
double Intermediate_206 = pow(Intermediate_205,Intermediate_80);
double Intermediate_207 = Intermediate_35+Intermediate_206+Intermediate_204+Intermediate_202+Intermediate_200;
double Intermediate_208 = Intermediate_201*Intermediate_61;
double Intermediate_209 = Intermediate_203*Intermediate_69;
double Intermediate_210 = Intermediate_205*Intermediate_75;
double Intermediate_211 = Intermediate_210+Intermediate_209+Intermediate_208;
double Intermediate_212 = Intermediate_196*Intermediate_211*Intermediate_207*Intermediate_188;
double Intermediate_213 = Intermediate_23*Intermediate_196*Intermediate_201*Intermediate_188;
double Intermediate_214 = Intermediate_162*Intermediate_167*Intermediate_151;
double Intermediate_215 = Intermediate_214+Intermediate_213;
double Intermediate_216 = Intermediate_23*Intermediate_215*Intermediate_61;
double Intermediate_217 = Intermediate_23*Intermediate_196*Intermediate_203*Intermediate_188;
double Intermediate_218 = Intermediate_162*Intermediate_169*Intermediate_151;
double Intermediate_219 = Intermediate_218+Intermediate_217;
double Intermediate_220 = Intermediate_23*Intermediate_219*Intermediate_69;
double Intermediate_221 = Intermediate_23*Intermediate_196*Intermediate_205*Intermediate_188;
double Intermediate_222 = Intermediate_162*Intermediate_171*Intermediate_151;
double Intermediate_223 = Intermediate_222+Intermediate_221;
double Intermediate_224 = Intermediate_23*Intermediate_223*Intermediate_75;
double Intermediate_225 = 0.5;
double Intermediate_226 = pow(Intermediate_163,Intermediate_225);
double Intermediate_227 = Intermediate_226*Intermediate_167;
double Intermediate_228 = pow(Intermediate_197,Intermediate_225);
double Intermediate_229 = Intermediate_228*Intermediate_201;
double Intermediate_230 = Intermediate_229+Intermediate_227;
double Intermediate_231 = Intermediate_228+Intermediate_226;
double Intermediate_232 = pow(Intermediate_231,Intermediate_23);
double Intermediate_233 = Intermediate_232*Intermediate_230*Intermediate_61;
double Intermediate_234 = Intermediate_226*Intermediate_169;
double Intermediate_235 = Intermediate_228*Intermediate_203;
double Intermediate_236 = Intermediate_235+Intermediate_234;
double Intermediate_237 = Intermediate_232*Intermediate_236*Intermediate_69;
double Intermediate_238 = Intermediate_226*Intermediate_171;
double Intermediate_239 = Intermediate_228*Intermediate_205;
double Intermediate_240 = Intermediate_239+Intermediate_238;
double Intermediate_241 = Intermediate_232*Intermediate_240*Intermediate_75;
double Intermediate_242 = Intermediate_241+Intermediate_237+Intermediate_233;
double Intermediate_243 = Intermediate_23*Intermediate_196*Intermediate_188;
double Intermediate_244 = Intermediate_163+Intermediate_243;
double Intermediate_245 = Intermediate_244*Intermediate_242;
double Intermediate_246 = Intermediate_245+Intermediate_224+Intermediate_220+Intermediate_216;

double Intermediate_248 = pow(Intermediate_230,Intermediate_80);
double Intermediate_249 = -2;
double Intermediate_250 = pow(Intermediate_231,Intermediate_249);
double Intermediate_251 = Intermediate_23*Intermediate_250*Intermediate_248;
double Intermediate_252 = pow(Intermediate_236,Intermediate_80);
double Intermediate_253 = Intermediate_23*Intermediate_250*Intermediate_252;
double Intermediate_254 = pow(Intermediate_240,Intermediate_80);
double Intermediate_255 = Intermediate_23*Intermediate_250*Intermediate_254;
double Intermediate_256 = Intermediate_226*Intermediate_173;
double Intermediate_257 = Intermediate_228*Intermediate_207;
double Intermediate_258 = Intermediate_257+Intermediate_256;
double Intermediate_259 = Intermediate_232*Intermediate_258;
double Intermediate_260 = -0.1;
double Intermediate_261 = Intermediate_260+Intermediate_259+Intermediate_255+Intermediate_253+Intermediate_251;
double Intermediate_262 = pow(Intermediate_261,Intermediate_225);
double Intermediate_263 = Intermediate_262+Intermediate_241+Intermediate_237+Intermediate_233;

int Intermediate_265 = Intermediate_263 < Intermediate_0;
double Intermediate_266 = Intermediate_23*Intermediate_232*Intermediate_230*Intermediate_61;
double Intermediate_267 = Intermediate_23*Intermediate_232*Intermediate_236*Intermediate_69;
double Intermediate_268 = Intermediate_23*Intermediate_232*Intermediate_240*Intermediate_75;
double Intermediate_269 = Intermediate_23*Intermediate_262;
double Intermediate_270 = Intermediate_269+Intermediate_268+Intermediate_267+Intermediate_266;


                double Intermediate_272;
                if (Intermediate_265) 
                    Intermediate_272 = Intermediate_270;
                else 
                    Intermediate_272 = Intermediate_263;
                

double Intermediate_274 = Intermediate_23*Intermediate_167*Intermediate_61;
double Intermediate_275 = Intermediate_23*Intermediate_169*Intermediate_69;
double Intermediate_276 = Intermediate_23*Intermediate_171*Intermediate_75;
double Intermediate_277 = Intermediate_210+Intermediate_209+Intermediate_208+Intermediate_276+Intermediate_275+Intermediate_274;

int Intermediate_279 = Intermediate_277 < Intermediate_0;
double Intermediate_280 = Intermediate_23*Intermediate_201*Intermediate_61;
double Intermediate_281 = Intermediate_23*Intermediate_203*Intermediate_69;
double Intermediate_282 = Intermediate_23*Intermediate_205*Intermediate_75;
double Intermediate_283 = Intermediate_176+Intermediate_175+Intermediate_174+Intermediate_282+Intermediate_281+Intermediate_280;


                double Intermediate_285;
                if (Intermediate_279) 
                    Intermediate_285 = Intermediate_283;
                else 
                    Intermediate_285 = Intermediate_277;
                
double Intermediate_286 = 2.0;
double Intermediate_287 = Intermediate_286*Intermediate_285;
double Intermediate_288 = pow(Intermediate_151,Intermediate_23);
double Intermediate_289 = Intermediate_288*Intermediate_153*Intermediate_161;
double Intermediate_290 = pow(Intermediate_289,Intermediate_225);
double Intermediate_291 = Intermediate_23*Intermediate_290;
double Intermediate_292 = pow(Intermediate_188,Intermediate_23);
double Intermediate_293 = Intermediate_292*Intermediate_189*Intermediate_195;
double Intermediate_294 = pow(Intermediate_293,Intermediate_225);
double Intermediate_295 = Intermediate_294+Intermediate_291;

int Intermediate_297 = Intermediate_295 < Intermediate_0;
double Intermediate_298 = Intermediate_23*Intermediate_294;
double Intermediate_299 = Intermediate_290+Intermediate_298;


                double Intermediate_301;
                if (Intermediate_297) 
                    Intermediate_301 = Intermediate_299;
                else 
                    Intermediate_301 = Intermediate_295;
                
double Intermediate_302 = Intermediate_286*Intermediate_301;
double Intermediate_303 = Intermediate_286+Intermediate_302+Intermediate_287;
int Intermediate_304 = Intermediate_272 < Intermediate_303;
double Intermediate_305 = 0.25;
double Intermediate_306 = Intermediate_305+Intermediate_272;
double Intermediate_307 = Intermediate_72+Intermediate_301+Intermediate_285;
double Intermediate_308 = pow(Intermediate_307,Intermediate_23);
double Intermediate_309 = Intermediate_308*Intermediate_306*Intermediate_272;
double Intermediate_310 = Intermediate_72+Intermediate_309+Intermediate_301+Intermediate_285;


                double Intermediate_312;
                if (Intermediate_304) 
                    Intermediate_312 = Intermediate_310;
                else 
                    Intermediate_312 = Intermediate_272;
                
double Intermediate_313 = Intermediate_269+Intermediate_241+Intermediate_237+Intermediate_233;

int Intermediate_315 = Intermediate_313 < Intermediate_0;
double Intermediate_316 = Intermediate_262+Intermediate_268+Intermediate_267+Intermediate_266;


                double Intermediate_318;
                if (Intermediate_315) 
                    Intermediate_318 = Intermediate_316;
                else 
                    Intermediate_318 = Intermediate_313;
                

int Intermediate_320 = Intermediate_318 < Intermediate_303;
double Intermediate_321 = Intermediate_305+Intermediate_318;
double Intermediate_322 = Intermediate_308*Intermediate_321*Intermediate_318;
double Intermediate_323 = Intermediate_72+Intermediate_322+Intermediate_301+Intermediate_285;


                double Intermediate_325;
                if (Intermediate_320) 
                    Intermediate_325 = Intermediate_323;
                else 
                    Intermediate_325 = Intermediate_318;
                
double Intermediate_326 = Intermediate_23*Intermediate_325;
double Intermediate_327 = Intermediate_35+Intermediate_326+Intermediate_312;
double Intermediate_328 = -0.5;
double Intermediate_329 = pow(Intermediate_261,Intermediate_328);
double Intermediate_330 = Intermediate_23*Intermediate_329*Intermediate_327*Intermediate_246;
double Intermediate_331 = Intermediate_23*Intermediate_196*Intermediate_207*Intermediate_188;
double Intermediate_332 = Intermediate_23*Intermediate_232*Intermediate_230*Intermediate_215;
double Intermediate_333 = Intermediate_23*Intermediate_232*Intermediate_236*Intermediate_219;
double Intermediate_334 = Intermediate_23*Intermediate_232*Intermediate_240*Intermediate_223;
double Intermediate_335 = Intermediate_162*Intermediate_173*Intermediate_151;
double Intermediate_336 = Intermediate_23*Intermediate_5*Intermediate_141;
double Intermediate_337 = Intermediate_23*Intermediate_8*Intermediate_143;
double Intermediate_338 = Intermediate_23*Intermediate_11*Intermediate_145;
double Intermediate_339 = Intermediate_23*Intermediate_149*Intermediate_22;
double Intermediate_340 = Intermediate_250*Intermediate_248;
double Intermediate_341 = Intermediate_250*Intermediate_252;
double Intermediate_342 = Intermediate_250*Intermediate_254;
double Intermediate_343 = Intermediate_35+Intermediate_342+Intermediate_341+Intermediate_340;
double Intermediate_344 = Intermediate_244*Intermediate_343;
double Intermediate_345 = Intermediate_154+Intermediate_148+Intermediate_187+Intermediate_344+Intermediate_184+Intermediate_182+Intermediate_180+Intermediate_339+Intermediate_338+Intermediate_337+Intermediate_336+Intermediate_335+Intermediate_334+Intermediate_333+Intermediate_332+Intermediate_331+Intermediate_147;

int Intermediate_347 = Intermediate_242 < Intermediate_0;
double Intermediate_348 = Intermediate_268+Intermediate_267+Intermediate_266;


                double Intermediate_350;
                if (Intermediate_347) 
                    Intermediate_350 = Intermediate_348;
                else 
                    Intermediate_350 = Intermediate_242;
                

int Intermediate_352 = Intermediate_350 < Intermediate_303;
double Intermediate_353 = Intermediate_305+Intermediate_350;
double Intermediate_354 = Intermediate_308*Intermediate_353*Intermediate_350;
double Intermediate_355 = Intermediate_72+Intermediate_354+Intermediate_301+Intermediate_285;


                double Intermediate_357;
                if (Intermediate_352) 
                    Intermediate_357 = Intermediate_355;
                else 
                    Intermediate_357 = Intermediate_350;
                
double Intermediate_358 = Intermediate_23*Intermediate_357;
double Intermediate_359 = Intermediate_35+Intermediate_358+Intermediate_325+Intermediate_312;
double Intermediate_360 = pow(Intermediate_261,Intermediate_23);
double Intermediate_361 = Intermediate_360*Intermediate_359*Intermediate_345;
double Intermediate_362 = Intermediate_361+Intermediate_330;
double Intermediate_363 = Intermediate_23*Intermediate_232*Intermediate_258*Intermediate_362;
double Intermediate_364 = Intermediate_148+Intermediate_187+Intermediate_184+Intermediate_182+Intermediate_180+Intermediate_339+Intermediate_338+Intermediate_337+Intermediate_336+Intermediate_335+Intermediate_331+Intermediate_147;
double Intermediate_365 = Intermediate_23*Intermediate_364*Intermediate_357;
double Intermediate_366 = Intermediate_23*Intermediate_329*Intermediate_327*Intermediate_345;
double Intermediate_367 = Intermediate_359*Intermediate_246;
double Intermediate_368 = Intermediate_367+Intermediate_366;
double Intermediate_369 = Intermediate_368*Intermediate_242;
double Intermediate_370 = Intermediate_369+Intermediate_365+Intermediate_363+Intermediate_212+Intermediate_178+Intermediate_139+Intermediate_115+Intermediate_86+Intermediate_39;
double Intermediate_371 = *(Tensor_1 + i*1 + 0);
double Intermediate_372 = pow(Intermediate_371,Intermediate_23);
double Intermediate_373 = Intermediate_372*Intermediate_370*Intermediate_1; *(Tensor_803 + i*1 + 0) = Intermediate_373;
double Intermediate_374 = Intermediate_162*Intermediate_177*Intermediate_167*Intermediate_151;
double Intermediate_375 = Intermediate_196*Intermediate_211*Intermediate_201*Intermediate_188;
double Intermediate_376 = Intermediate_23*Intermediate_37*Intermediate_85*Intermediate_32;
double Intermediate_377 = Intermediate_23*Intermediate_232*Intermediate_230*Intermediate_362;
double Intermediate_378 = Intermediate_23*Intermediate_215*Intermediate_357;
double Intermediate_379 = Intermediate_187+Intermediate_150+Intermediate_184+Intermediate_182+Intermediate_180+Intermediate_146+Intermediate_144+Intermediate_142+Intermediate_147+Intermediate_140;
double Intermediate_380 = Intermediate_379*Intermediate_61;
double Intermediate_381 = Intermediate_368*Intermediate_61;
double Intermediate_382 = Intermediate_381+Intermediate_380+Intermediate_378+Intermediate_377+Intermediate_376+Intermediate_375+Intermediate_374;
double Intermediate_383 = Intermediate_372*Intermediate_382*Intermediate_1; *(Tensor_798 + i*3 + 2) = Intermediate_383;
double Intermediate_384 = Intermediate_162*Intermediate_177*Intermediate_169*Intermediate_151;
double Intermediate_385 = Intermediate_196*Intermediate_211*Intermediate_203*Intermediate_188;
double Intermediate_386 = Intermediate_23*Intermediate_37*Intermediate_114*Intermediate_32;
double Intermediate_387 = Intermediate_23*Intermediate_232*Intermediate_236*Intermediate_362;
double Intermediate_388 = Intermediate_23*Intermediate_219*Intermediate_357;
double Intermediate_389 = Intermediate_379*Intermediate_69;
double Intermediate_390 = Intermediate_368*Intermediate_69;
double Intermediate_391 = Intermediate_390+Intermediate_389+Intermediate_388+Intermediate_387+Intermediate_386+Intermediate_385+Intermediate_384;
double Intermediate_392 = Intermediate_372*Intermediate_391*Intermediate_1; *(Tensor_798 + i*3 + 1) = Intermediate_392;
double Intermediate_393 = Intermediate_162*Intermediate_177*Intermediate_171*Intermediate_151;
double Intermediate_394 = Intermediate_196*Intermediate_211*Intermediate_205*Intermediate_188;
double Intermediate_395 = Intermediate_23*Intermediate_37*Intermediate_138*Intermediate_32;
double Intermediate_396 = Intermediate_23*Intermediate_232*Intermediate_240*Intermediate_362;
double Intermediate_397 = Intermediate_23*Intermediate_223*Intermediate_357;
double Intermediate_398 = Intermediate_379*Intermediate_75;
double Intermediate_399 = Intermediate_368*Intermediate_75;
double Intermediate_400 = Intermediate_399+Intermediate_398+Intermediate_397+Intermediate_396+Intermediate_395+Intermediate_394+Intermediate_393;
double Intermediate_401 = Intermediate_372*Intermediate_400*Intermediate_1; *(Tensor_798 + i*3 + 0) = Intermediate_401;
double Intermediate_402 = Intermediate_23*Intermediate_360*Intermediate_359*Intermediate_345;
double Intermediate_403 = Intermediate_162*Intermediate_177*Intermediate_151;
double Intermediate_404 = Intermediate_196*Intermediate_211*Intermediate_188;
double Intermediate_405 = Intermediate_329*Intermediate_327*Intermediate_246;
double Intermediate_406 = Intermediate_23*Intermediate_244*Intermediate_357;
double Intermediate_407 = Intermediate_406+Intermediate_405+Intermediate_404+Intermediate_403+Intermediate_402;
double Intermediate_408 = Intermediate_372*Intermediate_407*Intermediate_1; *(Tensor_793 + i*1 + 0) = Intermediate_408;
}
long long end = current_timestamp(); mil += end-start; printf("c module: %lld\n", mil);}
void Function_coupledFlux(int n, double* Tensor_805, double* Tensor_806, double* Tensor_807, double* Tensor_808, double* Tensor_809, double* Tensor_810, double* Tensor_811, double* Tensor_812, double* Tensor_813, double* Tensor_814, double* Tensor_815, double* Tensor_816, double* Tensor_0, double* Tensor_1, double* Tensor_2, double* Tensor_3, double* Tensor_4, double* Tensor_5, double* Tensor_6, double* Tensor_7, double* Tensor_1190, double* Tensor_1191, double* Tensor_1195, double* Tensor_1197, double* Tensor_1200, double* Tensor_1201) {
long long start = current_timestamp();for (int i = 0; i < n; i++) {
double Intermediate_0 = 0; *(Tensor_1201 + i*1 + 0) = Intermediate_0;
double Intermediate_1 = *(Tensor_0 + i*1 + 0);
double Intermediate_2 = *(Tensor_808 + i*1 + 0);
double Intermediate_3 = *(Tensor_807 + i*1 + 0);
double Intermediate_4 = *(Tensor_7 + i*6 + 5);
double Intermediate_5 = *(Tensor_814 + i*3 + 2);
double Intermediate_6 = Intermediate_5*Intermediate_4;
double Intermediate_7 = *(Tensor_7 + i*6 + 4);
double Intermediate_8 = *(Tensor_814 + i*3 + 1);
double Intermediate_9 = Intermediate_8*Intermediate_7;
double Intermediate_10 = *(Tensor_7 + i*6 + 3);
double Intermediate_11 = *(Tensor_814 + i*3 + 0);
double Intermediate_12 = Intermediate_11*Intermediate_10;
double Intermediate_13 = *(Tensor_813 + i*3 + 2);
double Intermediate_14 = *(Tensor_7 + i*6 + 2);
double Intermediate_15 = Intermediate_14*Intermediate_13;
double Intermediate_16 = *(Tensor_813 + i*3 + 1);
double Intermediate_17 = *(Tensor_7 + i*6 + 1);
double Intermediate_18 = Intermediate_17*Intermediate_16;
double Intermediate_19 = *(Tensor_813 + i*3 + 0);
double Intermediate_20 = *(Tensor_7 + i*6 + 0);
double Intermediate_21 = Intermediate_20*Intermediate_19;
double Intermediate_22 = *(Tensor_6 + i*2 + 1);
double Intermediate_23 = -1;
double Intermediate_24 = Intermediate_23*Intermediate_2;
double Intermediate_25 = Intermediate_24+Intermediate_3;
double Intermediate_26 = Intermediate_25*Intermediate_22;
double Intermediate_27 = *(Tensor_6 + i*2 + 0);
double Intermediate_28 = Intermediate_23*Intermediate_3;
double Intermediate_29 = Intermediate_28+Intermediate_2;
double Intermediate_30 = Intermediate_29*Intermediate_27;
double Intermediate_31 = 0.500025;
double Intermediate_32 = Intermediate_31+Intermediate_30+Intermediate_26+Intermediate_21+Intermediate_18+Intermediate_15+Intermediate_12+Intermediate_9+Intermediate_6+Intermediate_3+Intermediate_2;
double Intermediate_33 = *(Tensor_4 + i*1 + 0);
double Intermediate_34 = pow(Intermediate_33,Intermediate_23);
double Intermediate_35 = 0.5;
double Intermediate_36 = Intermediate_35+Intermediate_30+Intermediate_26+Intermediate_21+Intermediate_18+Intermediate_15+Intermediate_12+Intermediate_9+Intermediate_6+Intermediate_3+Intermediate_2;
double Intermediate_37 = pow(Intermediate_36,Intermediate_23);
double Intermediate_38 = -1435.0;
double Intermediate_39 = Intermediate_38*Intermediate_37*Intermediate_34*Intermediate_29*Intermediate_32;
double Intermediate_40 = *(Tensor_805 + i*3 + 2);
double Intermediate_41 = *(Tensor_806 + i*3 + 2);
double Intermediate_42 = *(Tensor_812 + i*9 + 8);
double Intermediate_43 = Intermediate_42*Intermediate_4;
double Intermediate_44 = *(Tensor_812 + i*9 + 7);
double Intermediate_45 = Intermediate_44*Intermediate_7;
double Intermediate_46 = *(Tensor_812 + i*9 + 6);
double Intermediate_47 = Intermediate_46*Intermediate_10;
double Intermediate_48 = *(Tensor_811 + i*9 + 8);
double Intermediate_49 = Intermediate_48*Intermediate_14;
double Intermediate_50 = *(Tensor_811 + i*9 + 7);
double Intermediate_51 = Intermediate_17*Intermediate_50;
double Intermediate_52 = *(Tensor_811 + i*9 + 6);
double Intermediate_53 = Intermediate_20*Intermediate_52;
double Intermediate_54 = Intermediate_23*Intermediate_40;
double Intermediate_55 = Intermediate_54+Intermediate_41;
double Intermediate_56 = Intermediate_55*Intermediate_27;
double Intermediate_57 = Intermediate_23*Intermediate_41;
double Intermediate_58 = Intermediate_57+Intermediate_40;
double Intermediate_59 = Intermediate_58*Intermediate_22;
double Intermediate_60 = Intermediate_35+Intermediate_59+Intermediate_56+Intermediate_53+Intermediate_51+Intermediate_49+Intermediate_47+Intermediate_45+Intermediate_43+Intermediate_41+Intermediate_40;
double Intermediate_61 = *(Tensor_5 + i*3 + 2);
double Intermediate_62 = *(Tensor_812 + i*9 + 4);
double Intermediate_63 = *(Tensor_812 + i*9 + 0);
double Intermediate_64 = *(Tensor_811 + i*9 + 4);
double Intermediate_65 = *(Tensor_811 + i*9 + 0);
double Intermediate_66 = 2.16666666666667;
double Intermediate_67 = Intermediate_66+Intermediate_65+Intermediate_64+Intermediate_48+Intermediate_63+Intermediate_62+Intermediate_42;
double Intermediate_68 = Intermediate_23*Intermediate_67*Intermediate_61;
double Intermediate_69 = *(Tensor_5 + i*3 + 1);
double Intermediate_70 = *(Tensor_812 + i*9 + 5);
double Intermediate_71 = *(Tensor_811 + i*9 + 5);
double Intermediate_72 = 1.0;
double Intermediate_73 = Intermediate_72+Intermediate_71+Intermediate_50+Intermediate_70+Intermediate_44;
double Intermediate_74 = Intermediate_73*Intermediate_69;
double Intermediate_75 = *(Tensor_5 + i*3 + 0);
double Intermediate_76 = *(Tensor_812 + i*9 + 2);
double Intermediate_77 = *(Tensor_811 + i*9 + 2);
double Intermediate_78 = Intermediate_72+Intermediate_77+Intermediate_52+Intermediate_76+Intermediate_46;
double Intermediate_79 = Intermediate_78*Intermediate_75;
double Intermediate_80 = 2;
double Intermediate_81 = Intermediate_80*Intermediate_42;
double Intermediate_82 = Intermediate_80*Intermediate_48;
double Intermediate_83 = Intermediate_72+Intermediate_82+Intermediate_81;
double Intermediate_84 = Intermediate_83*Intermediate_61;
double Intermediate_85 = Intermediate_84+Intermediate_79+Intermediate_74+Intermediate_68;
double Intermediate_86 = Intermediate_23*Intermediate_37*Intermediate_85*Intermediate_60*Intermediate_32;
double Intermediate_87 = *(Tensor_805 + i*3 + 1);
double Intermediate_88 = *(Tensor_806 + i*3 + 1);
double Intermediate_89 = Intermediate_70*Intermediate_4;
double Intermediate_90 = Intermediate_62*Intermediate_7;
double Intermediate_91 = *(Tensor_812 + i*9 + 3);
double Intermediate_92 = Intermediate_91*Intermediate_10;
double Intermediate_93 = Intermediate_71*Intermediate_14;
double Intermediate_94 = Intermediate_17*Intermediate_64;
double Intermediate_95 = *(Tensor_811 + i*9 + 3);
double Intermediate_96 = Intermediate_20*Intermediate_95;
double Intermediate_97 = Intermediate_23*Intermediate_87;
double Intermediate_98 = Intermediate_97+Intermediate_88;
double Intermediate_99 = Intermediate_98*Intermediate_27;
double Intermediate_100 = Intermediate_23*Intermediate_88;
double Intermediate_101 = Intermediate_100+Intermediate_87;
double Intermediate_102 = Intermediate_101*Intermediate_22;
double Intermediate_103 = Intermediate_35+Intermediate_102+Intermediate_99+Intermediate_96+Intermediate_94+Intermediate_93+Intermediate_92+Intermediate_90+Intermediate_89+Intermediate_88+Intermediate_87;
double Intermediate_104 = Intermediate_23*Intermediate_67*Intermediate_69;
double Intermediate_105 = Intermediate_73*Intermediate_61;
double Intermediate_106 = *(Tensor_812 + i*9 + 1);
double Intermediate_107 = *(Tensor_811 + i*9 + 1);
double Intermediate_108 = Intermediate_72+Intermediate_107+Intermediate_95+Intermediate_106+Intermediate_91;
double Intermediate_109 = Intermediate_108*Intermediate_75;
double Intermediate_110 = Intermediate_80*Intermediate_62;
double Intermediate_111 = Intermediate_80*Intermediate_64;
double Intermediate_112 = Intermediate_72+Intermediate_111+Intermediate_110;
double Intermediate_113 = Intermediate_112*Intermediate_69;
double Intermediate_114 = Intermediate_113+Intermediate_109+Intermediate_105+Intermediate_104;
double Intermediate_115 = Intermediate_23*Intermediate_37*Intermediate_114*Intermediate_103*Intermediate_32;
double Intermediate_116 = *(Tensor_806 + i*3 + 0);
double Intermediate_117 = *(Tensor_805 + i*3 + 0);
double Intermediate_118 = Intermediate_76*Intermediate_4;
double Intermediate_119 = Intermediate_106*Intermediate_7;
double Intermediate_120 = Intermediate_63*Intermediate_10;
double Intermediate_121 = Intermediate_77*Intermediate_14;
double Intermediate_122 = Intermediate_107*Intermediate_17;
double Intermediate_123 = Intermediate_20*Intermediate_65;
double Intermediate_124 = Intermediate_23*Intermediate_116;
double Intermediate_125 = Intermediate_124+Intermediate_117;
double Intermediate_126 = Intermediate_125*Intermediate_22;
double Intermediate_127 = Intermediate_23*Intermediate_117;
double Intermediate_128 = Intermediate_127+Intermediate_116;
double Intermediate_129 = Intermediate_128*Intermediate_27;
double Intermediate_130 = Intermediate_35+Intermediate_129+Intermediate_126+Intermediate_123+Intermediate_122+Intermediate_121+Intermediate_120+Intermediate_119+Intermediate_118+Intermediate_117+Intermediate_116;
double Intermediate_131 = Intermediate_23*Intermediate_67*Intermediate_75;
double Intermediate_132 = Intermediate_78*Intermediate_61;
double Intermediate_133 = Intermediate_108*Intermediate_69;
double Intermediate_134 = Intermediate_80*Intermediate_63;
double Intermediate_135 = Intermediate_80*Intermediate_65;
double Intermediate_136 = Intermediate_72+Intermediate_135+Intermediate_134;
double Intermediate_137 = Intermediate_136*Intermediate_75;
double Intermediate_138 = Intermediate_137+Intermediate_133+Intermediate_132+Intermediate_131;
double Intermediate_139 = Intermediate_23*Intermediate_37*Intermediate_138*Intermediate_130*Intermediate_32;
double Intermediate_140 = *(Tensor_810 + i*1 + 0);
double Intermediate_141 = *(Tensor_816 + i*3 + 2);
double Intermediate_142 = Intermediate_141*Intermediate_4;
double Intermediate_143 = *(Tensor_816 + i*3 + 1);
double Intermediate_144 = Intermediate_143*Intermediate_7;
double Intermediate_145 = *(Tensor_816 + i*3 + 0);
double Intermediate_146 = Intermediate_10*Intermediate_145;
double Intermediate_147 = *(Tensor_809 + i*1 + 0);
double Intermediate_148 = Intermediate_23*Intermediate_140;
double Intermediate_149 = Intermediate_148+Intermediate_147;
double Intermediate_150 = Intermediate_149*Intermediate_22;
double Intermediate_151 = Intermediate_150+Intermediate_146+Intermediate_144+Intermediate_142+Intermediate_140;
double Intermediate_152 = 1.4;
double Intermediate_153 = Intermediate_152+Intermediate_150+Intermediate_146+Intermediate_144+Intermediate_142+Intermediate_140;
double Intermediate_154 = 0.4;
double Intermediate_155 = Intermediate_154*Intermediate_5*Intermediate_4;
double Intermediate_156 = Intermediate_154*Intermediate_8*Intermediate_7;
double Intermediate_157 = Intermediate_154*Intermediate_11*Intermediate_10;
double Intermediate_158 = Intermediate_154*Intermediate_25*Intermediate_22;
double Intermediate_159 = Intermediate_154*Intermediate_2;
double Intermediate_160 = 287.0;
double Intermediate_161 = Intermediate_160+Intermediate_159+Intermediate_158+Intermediate_157+Intermediate_156+Intermediate_155;
double Intermediate_162 = pow(Intermediate_161,Intermediate_23);
double Intermediate_163 = Intermediate_162*Intermediate_151;
double Intermediate_164 = Intermediate_154+Intermediate_163;
double Intermediate_165 = pow(Intermediate_164,Intermediate_23);
double Intermediate_166 = Intermediate_165*Intermediate_153;
double Intermediate_167 = Intermediate_126+Intermediate_120+Intermediate_119+Intermediate_118+Intermediate_116;
double Intermediate_168 = pow(Intermediate_167,Intermediate_80);
double Intermediate_169 = Intermediate_59+Intermediate_47+Intermediate_45+Intermediate_43+Intermediate_41;
double Intermediate_170 = pow(Intermediate_169,Intermediate_80);
double Intermediate_171 = Intermediate_102+Intermediate_92+Intermediate_90+Intermediate_89+Intermediate_88;
double Intermediate_172 = pow(Intermediate_171,Intermediate_80);
double Intermediate_173 = Intermediate_35+Intermediate_172+Intermediate_170+Intermediate_168+Intermediate_166;
double Intermediate_174 = Intermediate_167*Intermediate_75;
double Intermediate_175 = Intermediate_169*Intermediate_61;
double Intermediate_176 = Intermediate_171*Intermediate_69;
double Intermediate_177 = Intermediate_176+Intermediate_175+Intermediate_174;
double Intermediate_178 = Intermediate_162*Intermediate_177*Intermediate_173*Intermediate_151;
double Intermediate_179 = *(Tensor_815 + i*3 + 2);
double Intermediate_180 = Intermediate_14*Intermediate_179;
double Intermediate_181 = *(Tensor_815 + i*3 + 1);
double Intermediate_182 = Intermediate_17*Intermediate_181;
double Intermediate_183 = *(Tensor_815 + i*3 + 0);
double Intermediate_184 = Intermediate_20*Intermediate_183;
double Intermediate_185 = Intermediate_23*Intermediate_147;
double Intermediate_186 = Intermediate_185+Intermediate_140;
double Intermediate_187 = Intermediate_186*Intermediate_27;
double Intermediate_188 = Intermediate_187+Intermediate_184+Intermediate_182+Intermediate_180+Intermediate_147;
double Intermediate_189 = Intermediate_152+Intermediate_187+Intermediate_184+Intermediate_182+Intermediate_180+Intermediate_147;
double Intermediate_190 = Intermediate_154*Intermediate_14*Intermediate_13;
double Intermediate_191 = Intermediate_154*Intermediate_17*Intermediate_16;
double Intermediate_192 = Intermediate_154*Intermediate_20*Intermediate_19;
double Intermediate_193 = Intermediate_154*Intermediate_29*Intermediate_27;
double Intermediate_194 = Intermediate_154*Intermediate_3;
double Intermediate_195 = Intermediate_160+Intermediate_194+Intermediate_193+Intermediate_192+Intermediate_191+Intermediate_190;
double Intermediate_196 = pow(Intermediate_195,Intermediate_23);
double Intermediate_197 = Intermediate_196*Intermediate_188;
double Intermediate_198 = Intermediate_154+Intermediate_197;
double Intermediate_199 = pow(Intermediate_198,Intermediate_23);
double Intermediate_200 = Intermediate_199*Intermediate_189;
double Intermediate_201 = Intermediate_56+Intermediate_53+Intermediate_51+Intermediate_49+Intermediate_40;
double Intermediate_202 = pow(Intermediate_201,Intermediate_80);
double Intermediate_203 = Intermediate_99+Intermediate_96+Intermediate_94+Intermediate_93+Intermediate_87;
double Intermediate_204 = pow(Intermediate_203,Intermediate_80);
double Intermediate_205 = Intermediate_129+Intermediate_123+Intermediate_122+Intermediate_121+Intermediate_117;
double Intermediate_206 = pow(Intermediate_205,Intermediate_80);
double Intermediate_207 = Intermediate_35+Intermediate_206+Intermediate_204+Intermediate_202+Intermediate_200;
double Intermediate_208 = Intermediate_201*Intermediate_61;
double Intermediate_209 = Intermediate_203*Intermediate_69;
double Intermediate_210 = Intermediate_205*Intermediate_75;
double Intermediate_211 = Intermediate_210+Intermediate_209+Intermediate_208;
double Intermediate_212 = Intermediate_196*Intermediate_211*Intermediate_207*Intermediate_188;
double Intermediate_213 = Intermediate_23*Intermediate_196*Intermediate_188*Intermediate_205;
double Intermediate_214 = Intermediate_162*Intermediate_151*Intermediate_167;
double Intermediate_215 = Intermediate_214+Intermediate_213;
double Intermediate_216 = Intermediate_23*Intermediate_215*Intermediate_75;
double Intermediate_217 = Intermediate_23*Intermediate_196*Intermediate_188*Intermediate_201;
double Intermediate_218 = Intermediate_162*Intermediate_169*Intermediate_151;
double Intermediate_219 = Intermediate_218+Intermediate_217;
double Intermediate_220 = Intermediate_23*Intermediate_219*Intermediate_61;
double Intermediate_221 = Intermediate_23*Intermediate_196*Intermediate_188*Intermediate_203;
double Intermediate_222 = Intermediate_162*Intermediate_171*Intermediate_151;
double Intermediate_223 = Intermediate_222+Intermediate_221;
double Intermediate_224 = Intermediate_23*Intermediate_223*Intermediate_69;
double Intermediate_225 = 0.5;
double Intermediate_226 = pow(Intermediate_163,Intermediate_225);
double Intermediate_227 = Intermediate_226*Intermediate_169;
double Intermediate_228 = pow(Intermediate_197,Intermediate_225);
double Intermediate_229 = Intermediate_228*Intermediate_201;
double Intermediate_230 = Intermediate_229+Intermediate_227;
double Intermediate_231 = Intermediate_228+Intermediate_226;
double Intermediate_232 = pow(Intermediate_231,Intermediate_23);
double Intermediate_233 = Intermediate_232*Intermediate_230*Intermediate_61;
double Intermediate_234 = Intermediate_226*Intermediate_171;
double Intermediate_235 = Intermediate_228*Intermediate_203;
double Intermediate_236 = Intermediate_235+Intermediate_234;
double Intermediate_237 = Intermediate_232*Intermediate_236*Intermediate_69;
double Intermediate_238 = Intermediate_226*Intermediate_167;
double Intermediate_239 = Intermediate_228*Intermediate_205;
double Intermediate_240 = Intermediate_239+Intermediate_238;
double Intermediate_241 = Intermediate_232*Intermediate_240*Intermediate_75;
double Intermediate_242 = Intermediate_241+Intermediate_237+Intermediate_233;
double Intermediate_243 = Intermediate_23*Intermediate_196*Intermediate_188;
double Intermediate_244 = Intermediate_163+Intermediate_243;
double Intermediate_245 = Intermediate_244*Intermediate_242;
double Intermediate_246 = Intermediate_245+Intermediate_224+Intermediate_220+Intermediate_216;

double Intermediate_248 = pow(Intermediate_230,Intermediate_80);
double Intermediate_249 = -2;
double Intermediate_250 = pow(Intermediate_231,Intermediate_249);
double Intermediate_251 = Intermediate_23*Intermediate_250*Intermediate_248;
double Intermediate_252 = pow(Intermediate_236,Intermediate_80);
double Intermediate_253 = Intermediate_23*Intermediate_250*Intermediate_252;
double Intermediate_254 = pow(Intermediate_240,Intermediate_80);
double Intermediate_255 = Intermediate_23*Intermediate_250*Intermediate_254;
double Intermediate_256 = Intermediate_226*Intermediate_173;
double Intermediate_257 = Intermediate_228*Intermediate_207;
double Intermediate_258 = Intermediate_257+Intermediate_256;
double Intermediate_259 = Intermediate_232*Intermediate_258;
double Intermediate_260 = -0.1;
double Intermediate_261 = Intermediate_260+Intermediate_259+Intermediate_255+Intermediate_253+Intermediate_251;
double Intermediate_262 = pow(Intermediate_261,Intermediate_225);
double Intermediate_263 = Intermediate_262+Intermediate_241+Intermediate_237+Intermediate_233;

int Intermediate_265 = Intermediate_263 < Intermediate_0;
double Intermediate_266 = Intermediate_23*Intermediate_232*Intermediate_230*Intermediate_61;
double Intermediate_267 = Intermediate_23*Intermediate_232*Intermediate_236*Intermediate_69;
double Intermediate_268 = Intermediate_23*Intermediate_232*Intermediate_240*Intermediate_75;
double Intermediate_269 = Intermediate_23*Intermediate_262;
double Intermediate_270 = Intermediate_269+Intermediate_268+Intermediate_267+Intermediate_266;


                double Intermediate_272;
                if (Intermediate_265) 
                    Intermediate_272 = Intermediate_270;
                else 
                    Intermediate_272 = Intermediate_263;
                

double Intermediate_274 = Intermediate_23*Intermediate_167*Intermediate_75;
double Intermediate_275 = Intermediate_23*Intermediate_169*Intermediate_61;
double Intermediate_276 = Intermediate_23*Intermediate_171*Intermediate_69;
double Intermediate_277 = Intermediate_210+Intermediate_209+Intermediate_208+Intermediate_276+Intermediate_275+Intermediate_274;

int Intermediate_279 = Intermediate_277 < Intermediate_0;
double Intermediate_280 = Intermediate_23*Intermediate_201*Intermediate_61;
double Intermediate_281 = Intermediate_23*Intermediate_203*Intermediate_69;
double Intermediate_282 = Intermediate_23*Intermediate_205*Intermediate_75;
double Intermediate_283 = Intermediate_176+Intermediate_175+Intermediate_174+Intermediate_282+Intermediate_281+Intermediate_280;


                double Intermediate_285;
                if (Intermediate_279) 
                    Intermediate_285 = Intermediate_283;
                else 
                    Intermediate_285 = Intermediate_277;
                
double Intermediate_286 = 2.0;
double Intermediate_287 = Intermediate_286*Intermediate_285;
double Intermediate_288 = pow(Intermediate_151,Intermediate_23);
double Intermediate_289 = Intermediate_288*Intermediate_153*Intermediate_161;
double Intermediate_290 = pow(Intermediate_289,Intermediate_225);
double Intermediate_291 = Intermediate_23*Intermediate_290;
double Intermediate_292 = pow(Intermediate_188,Intermediate_23);
double Intermediate_293 = Intermediate_292*Intermediate_189*Intermediate_195;
double Intermediate_294 = pow(Intermediate_293,Intermediate_225);
double Intermediate_295 = Intermediate_294+Intermediate_291;

int Intermediate_297 = Intermediate_295 < Intermediate_0;
double Intermediate_298 = Intermediate_23*Intermediate_294;
double Intermediate_299 = Intermediate_290+Intermediate_298;


                double Intermediate_301;
                if (Intermediate_297) 
                    Intermediate_301 = Intermediate_299;
                else 
                    Intermediate_301 = Intermediate_295;
                
double Intermediate_302 = Intermediate_286*Intermediate_301;
double Intermediate_303 = Intermediate_286+Intermediate_302+Intermediate_287;
int Intermediate_304 = Intermediate_272 < Intermediate_303;
double Intermediate_305 = 0.25;
double Intermediate_306 = Intermediate_305+Intermediate_272;
double Intermediate_307 = Intermediate_72+Intermediate_301+Intermediate_285;
double Intermediate_308 = pow(Intermediate_307,Intermediate_23);
double Intermediate_309 = Intermediate_308*Intermediate_306*Intermediate_272;
double Intermediate_310 = Intermediate_72+Intermediate_309+Intermediate_301+Intermediate_285;


                double Intermediate_312;
                if (Intermediate_304) 
                    Intermediate_312 = Intermediate_310;
                else 
                    Intermediate_312 = Intermediate_272;
                
double Intermediate_313 = Intermediate_269+Intermediate_241+Intermediate_237+Intermediate_233;

int Intermediate_315 = Intermediate_313 < Intermediate_0;
double Intermediate_316 = Intermediate_262+Intermediate_268+Intermediate_267+Intermediate_266;


                double Intermediate_318;
                if (Intermediate_315) 
                    Intermediate_318 = Intermediate_316;
                else 
                    Intermediate_318 = Intermediate_313;
                

int Intermediate_320 = Intermediate_318 < Intermediate_303;
double Intermediate_321 = Intermediate_305+Intermediate_318;
double Intermediate_322 = Intermediate_308*Intermediate_321*Intermediate_318;
double Intermediate_323 = Intermediate_72+Intermediate_322+Intermediate_301+Intermediate_285;


                double Intermediate_325;
                if (Intermediate_320) 
                    Intermediate_325 = Intermediate_323;
                else 
                    Intermediate_325 = Intermediate_318;
                
double Intermediate_326 = Intermediate_23*Intermediate_325;
double Intermediate_327 = Intermediate_35+Intermediate_326+Intermediate_312;
double Intermediate_328 = -0.5;
double Intermediate_329 = pow(Intermediate_261,Intermediate_328);
double Intermediate_330 = Intermediate_23*Intermediate_329*Intermediate_327*Intermediate_246;
double Intermediate_331 = Intermediate_23*Intermediate_196*Intermediate_207*Intermediate_188;
double Intermediate_332 = Intermediate_23*Intermediate_232*Intermediate_230*Intermediate_219;
double Intermediate_333 = Intermediate_23*Intermediate_232*Intermediate_236*Intermediate_223;
double Intermediate_334 = Intermediate_23*Intermediate_232*Intermediate_240*Intermediate_215;
double Intermediate_335 = Intermediate_162*Intermediate_173*Intermediate_151;
double Intermediate_336 = Intermediate_23*Intermediate_141*Intermediate_4;
double Intermediate_337 = Intermediate_23*Intermediate_143*Intermediate_7;
double Intermediate_338 = Intermediate_23*Intermediate_10*Intermediate_145;
double Intermediate_339 = Intermediate_23*Intermediate_149*Intermediate_22;
double Intermediate_340 = Intermediate_250*Intermediate_248;
double Intermediate_341 = Intermediate_250*Intermediate_252;
double Intermediate_342 = Intermediate_250*Intermediate_254;
double Intermediate_343 = Intermediate_35+Intermediate_342+Intermediate_341+Intermediate_340;
double Intermediate_344 = Intermediate_244*Intermediate_343;
double Intermediate_345 = Intermediate_154+Intermediate_148+Intermediate_187+Intermediate_344+Intermediate_184+Intermediate_182+Intermediate_180+Intermediate_339+Intermediate_338+Intermediate_337+Intermediate_336+Intermediate_335+Intermediate_334+Intermediate_333+Intermediate_332+Intermediate_331+Intermediate_147;

int Intermediate_347 = Intermediate_242 < Intermediate_0;
double Intermediate_348 = Intermediate_268+Intermediate_267+Intermediate_266;


                double Intermediate_350;
                if (Intermediate_347) 
                    Intermediate_350 = Intermediate_348;
                else 
                    Intermediate_350 = Intermediate_242;
                

int Intermediate_352 = Intermediate_350 < Intermediate_303;
double Intermediate_353 = Intermediate_305+Intermediate_350;
double Intermediate_354 = Intermediate_308*Intermediate_353*Intermediate_350;
double Intermediate_355 = Intermediate_72+Intermediate_354+Intermediate_301+Intermediate_285;


                double Intermediate_357;
                if (Intermediate_352) 
                    Intermediate_357 = Intermediate_355;
                else 
                    Intermediate_357 = Intermediate_350;
                
double Intermediate_358 = Intermediate_23*Intermediate_357;
double Intermediate_359 = Intermediate_35+Intermediate_358+Intermediate_325+Intermediate_312;
double Intermediate_360 = pow(Intermediate_261,Intermediate_23);
double Intermediate_361 = Intermediate_360*Intermediate_359*Intermediate_345;
double Intermediate_362 = Intermediate_361+Intermediate_330;
double Intermediate_363 = Intermediate_23*Intermediate_232*Intermediate_258*Intermediate_362;
double Intermediate_364 = Intermediate_148+Intermediate_187+Intermediate_184+Intermediate_182+Intermediate_180+Intermediate_339+Intermediate_338+Intermediate_337+Intermediate_336+Intermediate_335+Intermediate_331+Intermediate_147;
double Intermediate_365 = Intermediate_23*Intermediate_364*Intermediate_357;
double Intermediate_366 = Intermediate_23*Intermediate_329*Intermediate_327*Intermediate_345;
double Intermediate_367 = Intermediate_359*Intermediate_246;
double Intermediate_368 = Intermediate_367+Intermediate_366;
double Intermediate_369 = Intermediate_368*Intermediate_242;
double Intermediate_370 = Intermediate_369+Intermediate_365+Intermediate_363+Intermediate_212+Intermediate_178+Intermediate_139+Intermediate_115+Intermediate_86+Intermediate_39;
double Intermediate_371 = *(Tensor_1 + i*1 + 0);
double Intermediate_372 = pow(Intermediate_371,Intermediate_23);
double Intermediate_373 = Intermediate_372*Intermediate_370*Intermediate_1; *(Tensor_1200 + i*1 + 0) = Intermediate_373;
double Intermediate_374 = Intermediate_162*Intermediate_177*Intermediate_169*Intermediate_151;
double Intermediate_375 = Intermediate_196*Intermediate_211*Intermediate_188*Intermediate_201;
double Intermediate_376 = Intermediate_23*Intermediate_37*Intermediate_85*Intermediate_32;
double Intermediate_377 = Intermediate_23*Intermediate_232*Intermediate_230*Intermediate_362;
double Intermediate_378 = Intermediate_23*Intermediate_219*Intermediate_357;
double Intermediate_379 = Intermediate_187+Intermediate_150+Intermediate_184+Intermediate_182+Intermediate_180+Intermediate_146+Intermediate_144+Intermediate_142+Intermediate_147+Intermediate_140;
double Intermediate_380 = Intermediate_379*Intermediate_61;
double Intermediate_381 = Intermediate_368*Intermediate_61;
double Intermediate_382 = Intermediate_381+Intermediate_380+Intermediate_378+Intermediate_377+Intermediate_376+Intermediate_375+Intermediate_374;
double Intermediate_383 = Intermediate_372*Intermediate_382*Intermediate_1; *(Tensor_1195 + i*3 + 2) = Intermediate_383;
double Intermediate_384 = Intermediate_162*Intermediate_177*Intermediate_171*Intermediate_151;
double Intermediate_385 = Intermediate_196*Intermediate_211*Intermediate_188*Intermediate_203;
double Intermediate_386 = Intermediate_23*Intermediate_37*Intermediate_114*Intermediate_32;
double Intermediate_387 = Intermediate_23*Intermediate_232*Intermediate_236*Intermediate_362;
double Intermediate_388 = Intermediate_23*Intermediate_223*Intermediate_357;
double Intermediate_389 = Intermediate_379*Intermediate_69;
double Intermediate_390 = Intermediate_368*Intermediate_69;
double Intermediate_391 = Intermediate_390+Intermediate_389+Intermediate_388+Intermediate_387+Intermediate_386+Intermediate_385+Intermediate_384;
double Intermediate_392 = Intermediate_372*Intermediate_391*Intermediate_1; *(Tensor_1195 + i*3 + 1) = Intermediate_392;
double Intermediate_393 = Intermediate_162*Intermediate_177*Intermediate_151*Intermediate_167;
double Intermediate_394 = Intermediate_196*Intermediate_211*Intermediate_188*Intermediate_205;
double Intermediate_395 = Intermediate_23*Intermediate_37*Intermediate_138*Intermediate_32;
double Intermediate_396 = Intermediate_23*Intermediate_232*Intermediate_240*Intermediate_362;
double Intermediate_397 = Intermediate_23*Intermediate_215*Intermediate_357;
double Intermediate_398 = Intermediate_379*Intermediate_75;
double Intermediate_399 = Intermediate_368*Intermediate_75;
double Intermediate_400 = Intermediate_399+Intermediate_398+Intermediate_397+Intermediate_396+Intermediate_395+Intermediate_394+Intermediate_393;
double Intermediate_401 = Intermediate_372*Intermediate_400*Intermediate_1; *(Tensor_1195 + i*3 + 0) = Intermediate_401;
double Intermediate_402 = Intermediate_23*Intermediate_360*Intermediate_359*Intermediate_345;
double Intermediate_403 = Intermediate_162*Intermediate_177*Intermediate_151;
double Intermediate_404 = Intermediate_196*Intermediate_211*Intermediate_188;
double Intermediate_405 = Intermediate_329*Intermediate_327*Intermediate_246;
double Intermediate_406 = Intermediate_23*Intermediate_244*Intermediate_357;
double Intermediate_407 = Intermediate_406+Intermediate_405+Intermediate_404+Intermediate_403+Intermediate_402;
double Intermediate_408 = Intermediate_372*Intermediate_407*Intermediate_1; *(Tensor_1190 + i*1 + 0) = Intermediate_408;
}
long long end = current_timestamp(); mil += end-start; printf("c module: %lld\n", mil);}
void Function_boundaryFlux(int n, double* Tensor_1202, double* Tensor_1203, double* Tensor_1204, double* Tensor_1205, double* Tensor_1206, double* Tensor_1207, double* Tensor_1208, double* Tensor_1209, double* Tensor_1210, double* Tensor_1211, double* Tensor_1212, double* Tensor_1213, double* Tensor_0, double* Tensor_1, double* Tensor_2, double* Tensor_3, double* Tensor_4, double* Tensor_5, double* Tensor_6, double* Tensor_7, double* Tensor_1330, double* Tensor_1331, double* Tensor_1335, double* Tensor_1337, double* Tensor_1340, double* Tensor_1341) {
long long start = current_timestamp();for (int i = 0; i < n; i++) {
double Intermediate_0 = 0; *(Tensor_1341 + i*1 + 0) = Intermediate_0;
double Intermediate_1 = *(Tensor_0 + i*1 + 0);
double Intermediate_2 = *(Tensor_1205 + i*1 + 0);
double Intermediate_3 = *(Tensor_1204 + i*1 + 0);
double Intermediate_4 = -1;
double Intermediate_5 = Intermediate_4*Intermediate_3;
double Intermediate_6 = Intermediate_5+Intermediate_2;
double Intermediate_7 = 2.5e-5;
double Intermediate_8 = Intermediate_7+Intermediate_2;
double Intermediate_9 = *(Tensor_4 + i*1 + 0);
double Intermediate_10 = pow(Intermediate_9,Intermediate_4);
double Intermediate_11 = pow(Intermediate_2,Intermediate_4);
double Intermediate_12 = -1435.0;
double Intermediate_13 = Intermediate_12*Intermediate_11*Intermediate_10*Intermediate_8*Intermediate_6;
double Intermediate_14 = *(Tensor_1203 + i*3 + 2);
double Intermediate_15 = *(Tensor_5 + i*3 + 2);
double Intermediate_16 = *(Tensor_1209 + i*9 + 8);
double Intermediate_17 = 2;
double Intermediate_18 = Intermediate_17*Intermediate_16*Intermediate_15;
double Intermediate_19 = *(Tensor_1209 + i*9 + 4);
double Intermediate_20 = *(Tensor_1209 + i*9 + 0);
double Intermediate_21 = 0.666666666666667;
double Intermediate_22 = Intermediate_21+Intermediate_20+Intermediate_19+Intermediate_16;
double Intermediate_23 = Intermediate_4*Intermediate_22*Intermediate_15;
double Intermediate_24 = *(Tensor_5 + i*3 + 1);
double Intermediate_25 = *(Tensor_1209 + i*9 + 7);
double Intermediate_26 = *(Tensor_1209 + i*9 + 5);
double Intermediate_27 = Intermediate_26+Intermediate_25;
double Intermediate_28 = Intermediate_27*Intermediate_24;
double Intermediate_29 = *(Tensor_5 + i*3 + 0);
double Intermediate_30 = *(Tensor_1209 + i*9 + 6);
double Intermediate_31 = *(Tensor_1209 + i*9 + 2);
double Intermediate_32 = Intermediate_31+Intermediate_30;
double Intermediate_33 = Intermediate_32*Intermediate_29;
double Intermediate_34 = Intermediate_33+Intermediate_28+Intermediate_23+Intermediate_18;
double Intermediate_35 = Intermediate_4*Intermediate_11*Intermediate_8*Intermediate_34*Intermediate_14;
double Intermediate_36 = *(Tensor_1203 + i*3 + 0);
double Intermediate_37 = Intermediate_17*Intermediate_20*Intermediate_29;
double Intermediate_38 = Intermediate_4*Intermediate_22*Intermediate_29;
double Intermediate_39 = Intermediate_32*Intermediate_15;
double Intermediate_40 = *(Tensor_1209 + i*9 + 3);
double Intermediate_41 = *(Tensor_1209 + i*9 + 1);
double Intermediate_42 = Intermediate_41+Intermediate_40;
double Intermediate_43 = Intermediate_42*Intermediate_24;
double Intermediate_44 = Intermediate_43+Intermediate_39+Intermediate_38+Intermediate_37;
double Intermediate_45 = Intermediate_4*Intermediate_11*Intermediate_8*Intermediate_44*Intermediate_36;
double Intermediate_46 = *(Tensor_1203 + i*3 + 1);
double Intermediate_47 = Intermediate_17*Intermediate_19*Intermediate_24;
double Intermediate_48 = Intermediate_4*Intermediate_22*Intermediate_24;
double Intermediate_49 = Intermediate_27*Intermediate_15;
double Intermediate_50 = Intermediate_42*Intermediate_29;
double Intermediate_51 = Intermediate_50+Intermediate_49+Intermediate_48+Intermediate_47;
double Intermediate_52 = Intermediate_4*Intermediate_11*Intermediate_8*Intermediate_51*Intermediate_46;
double Intermediate_53 = Intermediate_14*Intermediate_15;
double Intermediate_54 = Intermediate_46*Intermediate_24;
double Intermediate_55 = Intermediate_36*Intermediate_29;
double Intermediate_56 = Intermediate_55+Intermediate_54+Intermediate_53;
double Intermediate_57 = *(Tensor_1207 + i*1 + 0);
double Intermediate_58 = pow(Intermediate_14,Intermediate_17);
double Intermediate_59 = pow(Intermediate_46,Intermediate_17);
double Intermediate_60 = pow(Intermediate_36,Intermediate_17);
double Intermediate_61 = 718.0;
double Intermediate_62 = Intermediate_61+Intermediate_60+Intermediate_59+Intermediate_58+Intermediate_2;
double Intermediate_63 = 0.4;
double Intermediate_64 = Intermediate_63*Intermediate_2;
double Intermediate_65 = 287.0;
double Intermediate_66 = Intermediate_65+Intermediate_64;
double Intermediate_67 = pow(Intermediate_66,Intermediate_4);
double Intermediate_68 = Intermediate_67*Intermediate_62*Intermediate_57;
double Intermediate_69 = Intermediate_68+Intermediate_57;
double Intermediate_70 = Intermediate_69*Intermediate_56;
double Intermediate_71 = Intermediate_70+Intermediate_52+Intermediate_45+Intermediate_35+Intermediate_13;
double Intermediate_72 = *(Tensor_1 + i*1 + 0);
double Intermediate_73 = pow(Intermediate_72,Intermediate_4);
double Intermediate_74 = Intermediate_73*Intermediate_71*Intermediate_1; *(Tensor_1340 + i*1 + 0) = Intermediate_74;
double Intermediate_75 = Intermediate_67*Intermediate_56*Intermediate_14*Intermediate_57;
double Intermediate_76 = Intermediate_4*Intermediate_11*Intermediate_8*Intermediate_34;
double Intermediate_77 = Intermediate_57*Intermediate_15;
double Intermediate_78 = Intermediate_77+Intermediate_76+Intermediate_75;
double Intermediate_79 = Intermediate_73*Intermediate_78*Intermediate_1; *(Tensor_1335 + i*3 + 2) = Intermediate_79;
double Intermediate_80 = Intermediate_67*Intermediate_56*Intermediate_46*Intermediate_57;
double Intermediate_81 = Intermediate_4*Intermediate_11*Intermediate_8*Intermediate_51;
double Intermediate_82 = Intermediate_57*Intermediate_24;
double Intermediate_83 = Intermediate_82+Intermediate_81+Intermediate_80;
double Intermediate_84 = Intermediate_73*Intermediate_83*Intermediate_1; *(Tensor_1335 + i*3 + 1) = Intermediate_84;
double Intermediate_85 = Intermediate_67*Intermediate_56*Intermediate_36*Intermediate_57;
double Intermediate_86 = Intermediate_4*Intermediate_11*Intermediate_8*Intermediate_44;
double Intermediate_87 = Intermediate_57*Intermediate_29;
double Intermediate_88 = Intermediate_87+Intermediate_86+Intermediate_85;
double Intermediate_89 = Intermediate_73*Intermediate_88*Intermediate_1; *(Tensor_1335 + i*3 + 0) = Intermediate_89;
double Intermediate_90 = Intermediate_67*Intermediate_73*Intermediate_56*Intermediate_1*Intermediate_57; *(Tensor_1330 + i*1 + 0) = Intermediate_90;
}
long long end = current_timestamp(); mil += end-start; printf("c module: %lld\n", mil);}