
void Function_primitive(int n, const scalar* Tensor_19, const scalar* Tensor_17, const scalar* Tensor_18, scalar* Tensor_20, scalar* Tensor_30, scalar* Tensor_28) {
	long long start = current_timestamp();
	for (integer i = 0; i < n; i++) {
		const scalar Intermediate_0 = -2;
		scalar Intermediate_1 = *(Tensor_19 + i*1 + 0);
		scalar Intermediate_2 = pow(Intermediate_1,Intermediate_0);
		const scalar Intermediate_3 = 2;
		scalar Intermediate_4 = *(Tensor_17 + i*3 + 2);
		scalar Intermediate_5 = pow(Intermediate_4,Intermediate_3);
		const scalar Intermediate_6 = -1;
		scalar Intermediate_7 = Intermediate_6*Intermediate_5*Intermediate_2;
		scalar Intermediate_8 = *(Tensor_17 + i*3 + 1);
		scalar Intermediate_9 = pow(Intermediate_8,Intermediate_3);
		scalar Intermediate_10 = Intermediate_6*Intermediate_9*Intermediate_2;
		scalar Intermediate_11 = *(Tensor_17 + i*3 + 0);
		scalar Intermediate_12 = pow(Intermediate_11,Intermediate_3);
		scalar Intermediate_13 = Intermediate_6*Intermediate_12*Intermediate_2;
		scalar Intermediate_14 = *(Tensor_18 + i*1 + 0);
		scalar Intermediate_15 = pow(Intermediate_1,Intermediate_6);
		scalar Intermediate_16 = Intermediate_15*Intermediate_14;
		const scalar Intermediate_17 = -0.5;
		scalar Intermediate_18 = Intermediate_17+Intermediate_16+Intermediate_13+Intermediate_10+Intermediate_7;
		const scalar Intermediate_19 = 0.4;
		scalar Intermediate_20 = Intermediate_19+Intermediate_1;
		scalar Intermediate_21 = Intermediate_20*Intermediate_18; *(Tensor_28 + i*1 + 0) = Intermediate_21;
		const scalar Intermediate_22 = -0.00139372822299652;
		scalar Intermediate_23 = Intermediate_22*Intermediate_5*Intermediate_2;
		scalar Intermediate_24 = Intermediate_22*Intermediate_9*Intermediate_2;
		scalar Intermediate_25 = Intermediate_22*Intermediate_12*Intermediate_2;
		const scalar Intermediate_26 = 0.00139372822299652;
		scalar Intermediate_27 = Intermediate_26*Intermediate_15*Intermediate_14;
		const scalar Intermediate_28 = -0.000696864111498258;
		scalar Intermediate_29 = Intermediate_28+Intermediate_27+Intermediate_25+Intermediate_24+Intermediate_23; *(Tensor_30 + i*1 + 0) = Intermediate_29;
		scalar Intermediate_30 = Intermediate_15*Intermediate_4; *(Tensor_20 + i*3 + 2) = Intermediate_30;
		scalar Intermediate_31 = Intermediate_15*Intermediate_8; *(Tensor_20 + i*3 + 1) = Intermediate_31;
		scalar Intermediate_32 = Intermediate_15*Intermediate_11; *(Tensor_20 + i*3 + 0) = Intermediate_32;
	}
	long long end = current_timestamp(); mil += end-start; printf("c module Function_primitive: %lld\n", mil);
}

void Function_grad(int n, const scalar* Tensor_31, const scalar* Tensor_32, const scalar* Tensor_33, const scalar* Tensor_0, const scalar* Tensor_1, const scalar* Tensor_2, const scalar* Tensor_3, const scalar* Tensor_4, const scalar* Tensor_5, const scalar* Tensor_6, const scalar* Tensor_7, const integer* Tensor_8, const integer* Tensor_9, scalar* Tensor_64, scalar* Tensor_71, scalar* Tensor_78) {
	long long start = current_timestamp();
	for (integer i = 0; i < n; i++) {
		integer Intermediate_0 = *(Tensor_9 + i*1 + 0);
		scalar Intermediate_1 = *(Tensor_5 + i*3 + 2);
		scalar Intermediate_2 = *(Tensor_0 + i*1 + 0);
		integer Intermediate_3 = *(Tensor_8 + i*1 + 0);
		
		scalar Intermediate_5 = *(Tensor_33 + Intermediate_3*1 + 0);
		scalar Intermediate_6 = *(Tensor_3 + i*1 + 0);
		scalar Intermediate_7 = Intermediate_6*Intermediate_5;
		integer Intermediate_8 = *(Tensor_9 + i*1 + 0);
		scalar Intermediate_9 = *(Tensor_33 + Intermediate_8*1 + 0);
		const scalar Intermediate_10 = -1;
		scalar Intermediate_11 = Intermediate_10*Intermediate_6;
		const scalar Intermediate_12 = 1;
		scalar Intermediate_13 = Intermediate_12+Intermediate_11;
		scalar Intermediate_14 = Intermediate_13*Intermediate_9;
		scalar Intermediate_15 = Intermediate_14+Intermediate_7;
		scalar Intermediate_16 = *(Tensor_2 + i*1 + 0);
		scalar Intermediate_17 = pow(Intermediate_16,Intermediate_10);
		scalar Intermediate_18 = Intermediate_10*Intermediate_17*Intermediate_15*Intermediate_2*Intermediate_1;
		scalar Intermediate_19 = *(Tensor_5 + i*3 + 1);
		scalar Intermediate_20 = Intermediate_10*Intermediate_17*Intermediate_15*Intermediate_2*Intermediate_19;
		scalar Intermediate_21 = *(Tensor_5 + i*3 + 0);
		scalar Intermediate_22 = Intermediate_10*Intermediate_17*Intermediate_15*Intermediate_2*Intermediate_21;
		scalar Intermediate_23 = *(Tensor_1 + i*1 + 0);
		scalar Intermediate_24 = pow(Intermediate_23,Intermediate_10);
		scalar Intermediate_25 = Intermediate_24*Intermediate_15*Intermediate_2*Intermediate_1;
		scalar Intermediate_26 = Intermediate_24*Intermediate_15*Intermediate_2*Intermediate_19;
		scalar Intermediate_27 = Intermediate_24*Intermediate_15*Intermediate_2*Intermediate_21;
		*(Tensor_78 + Intermediate_3*3 + 2) = Intermediate_27;
		*(Tensor_78 + Intermediate_3*3 + 2) = Intermediate_26;
		*(Tensor_78 + Intermediate_3*3 + 2) = Intermediate_25;
		*(Tensor_78 + Intermediate_8*3 + 2) = Intermediate_22;
		*(Tensor_78 + Intermediate_8*3 + 2) = Intermediate_20;
		*(Tensor_78 + Intermediate_8*3 + 2) = Intermediate_18;
		
		*(Tensor_78 + Intermediate_3*3 + 2) = Intermediate_27;
		*(Tensor_78 + Intermediate_3*3 + 2) = Intermediate_26;
		*(Tensor_78 + Intermediate_3*3 + 2) = Intermediate_25;
		*(Tensor_78 + Intermediate_8*3 + 2) = Intermediate_22;
		*(Tensor_78 + Intermediate_8*3 + 2) = Intermediate_20;
		*(Tensor_78 + Intermediate_8*3 + 2) = Intermediate_18;
		
		*(Tensor_78 + Intermediate_3*3 + 2) = Intermediate_27;
		*(Tensor_78 + Intermediate_3*3 + 2) = Intermediate_26;
		*(Tensor_78 + Intermediate_3*3 + 2) = Intermediate_25;
		*(Tensor_78 + Intermediate_8*3 + 2) = Intermediate_22;
		*(Tensor_78 + Intermediate_8*3 + 2) = Intermediate_20;
		*(Tensor_78 + Intermediate_8*3 + 2) = Intermediate_18;
		
		
		scalar Intermediate_32 = *(Tensor_32 + Intermediate_3*1 + 0);
		scalar Intermediate_33 = Intermediate_6*Intermediate_32;
		scalar Intermediate_34 = *(Tensor_32 + Intermediate_8*1 + 0);
		scalar Intermediate_35 = Intermediate_13*Intermediate_34;
		scalar Intermediate_36 = Intermediate_35+Intermediate_33;
		scalar Intermediate_37 = Intermediate_10*Intermediate_17*Intermediate_36*Intermediate_2*Intermediate_1;
		scalar Intermediate_38 = Intermediate_10*Intermediate_17*Intermediate_36*Intermediate_2*Intermediate_19;
		scalar Intermediate_39 = Intermediate_10*Intermediate_17*Intermediate_36*Intermediate_2*Intermediate_21;
		scalar Intermediate_40 = Intermediate_24*Intermediate_36*Intermediate_2*Intermediate_1;
		scalar Intermediate_41 = Intermediate_24*Intermediate_36*Intermediate_2*Intermediate_19;
		scalar Intermediate_42 = Intermediate_24*Intermediate_36*Intermediate_2*Intermediate_21;
		*(Tensor_71 + Intermediate_3*3 + 2) = Intermediate_42;
		*(Tensor_71 + Intermediate_3*3 + 2) = Intermediate_41;
		*(Tensor_71 + Intermediate_3*3 + 2) = Intermediate_40;
		*(Tensor_71 + Intermediate_8*3 + 2) = Intermediate_39;
		*(Tensor_71 + Intermediate_8*3 + 2) = Intermediate_38;
		*(Tensor_71 + Intermediate_8*3 + 2) = Intermediate_37;
		
		*(Tensor_71 + Intermediate_3*3 + 2) = Intermediate_42;
		*(Tensor_71 + Intermediate_3*3 + 2) = Intermediate_41;
		*(Tensor_71 + Intermediate_3*3 + 2) = Intermediate_40;
		*(Tensor_71 + Intermediate_8*3 + 2) = Intermediate_39;
		*(Tensor_71 + Intermediate_8*3 + 2) = Intermediate_38;
		*(Tensor_71 + Intermediate_8*3 + 2) = Intermediate_37;
		
		*(Tensor_71 + Intermediate_3*3 + 2) = Intermediate_42;
		*(Tensor_71 + Intermediate_3*3 + 2) = Intermediate_41;
		*(Tensor_71 + Intermediate_3*3 + 2) = Intermediate_40;
		*(Tensor_71 + Intermediate_8*3 + 2) = Intermediate_39;
		*(Tensor_71 + Intermediate_8*3 + 2) = Intermediate_38;
		*(Tensor_71 + Intermediate_8*3 + 2) = Intermediate_37;
		
		
		scalar Intermediate_47 = *(Tensor_31 + Intermediate_3*3 + 2);
		scalar Intermediate_48 = Intermediate_6*Intermediate_47;
		scalar Intermediate_49 = *(Tensor_31 + Intermediate_8*3 + 2);
		scalar Intermediate_50 = Intermediate_13*Intermediate_49;
		scalar Intermediate_51 = Intermediate_50+Intermediate_48;
		scalar Intermediate_52 = Intermediate_10*Intermediate_17*Intermediate_51*Intermediate_2*Intermediate_1;
		scalar Intermediate_53 = Intermediate_10*Intermediate_17*Intermediate_51*Intermediate_2*Intermediate_19;
		scalar Intermediate_54 = Intermediate_10*Intermediate_17*Intermediate_51*Intermediate_2*Intermediate_21;
		
		scalar Intermediate_56 = *(Tensor_31 + Intermediate_3*3 + 1);
		scalar Intermediate_57 = Intermediate_6*Intermediate_56;
		scalar Intermediate_58 = *(Tensor_31 + Intermediate_8*3 + 1);
		scalar Intermediate_59 = Intermediate_13*Intermediate_58;
		scalar Intermediate_60 = Intermediate_59+Intermediate_57;
		scalar Intermediate_61 = Intermediate_10*Intermediate_17*Intermediate_60*Intermediate_2*Intermediate_1;
		scalar Intermediate_62 = Intermediate_10*Intermediate_17*Intermediate_60*Intermediate_2*Intermediate_19;
		scalar Intermediate_63 = Intermediate_10*Intermediate_17*Intermediate_60*Intermediate_2*Intermediate_21;
		
		scalar Intermediate_65 = *(Tensor_31 + Intermediate_3*3 + 0);
		scalar Intermediate_66 = Intermediate_6*Intermediate_65;
		scalar Intermediate_67 = *(Tensor_31 + Intermediate_8*3 + 0);
		scalar Intermediate_68 = Intermediate_13*Intermediate_67;
		scalar Intermediate_69 = Intermediate_68+Intermediate_66;
		scalar Intermediate_70 = Intermediate_10*Intermediate_17*Intermediate_69*Intermediate_2*Intermediate_1;
		scalar Intermediate_71 = Intermediate_10*Intermediate_17*Intermediate_69*Intermediate_2*Intermediate_19;
		scalar Intermediate_72 = Intermediate_10*Intermediate_17*Intermediate_69*Intermediate_2*Intermediate_21;
		scalar Intermediate_73 = Intermediate_24*Intermediate_51*Intermediate_2*Intermediate_1;
		scalar Intermediate_74 = Intermediate_24*Intermediate_51*Intermediate_2*Intermediate_19;
		scalar Intermediate_75 = Intermediate_24*Intermediate_51*Intermediate_2*Intermediate_21;
		scalar Intermediate_76 = Intermediate_24*Intermediate_60*Intermediate_2*Intermediate_1;
		scalar Intermediate_77 = Intermediate_24*Intermediate_60*Intermediate_2*Intermediate_19;
		scalar Intermediate_78 = Intermediate_24*Intermediate_60*Intermediate_2*Intermediate_21;
		scalar Intermediate_79 = Intermediate_24*Intermediate_69*Intermediate_2*Intermediate_1;
		scalar Intermediate_80 = Intermediate_24*Intermediate_69*Intermediate_2*Intermediate_19;
		scalar Intermediate_81 = Intermediate_24*Intermediate_69*Intermediate_2*Intermediate_21;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_81;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_80;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_79;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_78;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_77;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_76;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_75;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_74;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_73;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_72;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_71;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_70;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_63;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_62;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_61;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_54;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_53;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_52;
		
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_81;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_80;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_79;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_78;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_77;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_76;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_75;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_74;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_73;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_72;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_71;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_70;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_63;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_62;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_61;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_54;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_53;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_52;
		
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_81;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_80;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_79;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_78;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_77;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_76;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_75;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_74;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_73;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_72;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_71;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_70;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_63;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_62;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_61;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_54;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_53;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_52;
		
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_81;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_80;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_79;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_78;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_77;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_76;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_75;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_74;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_73;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_72;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_71;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_70;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_63;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_62;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_61;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_54;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_53;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_52;
		
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_81;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_80;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_79;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_78;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_77;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_76;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_75;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_74;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_73;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_72;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_71;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_70;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_63;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_62;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_61;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_54;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_53;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_52;
		
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_81;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_80;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_79;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_78;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_77;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_76;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_75;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_74;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_73;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_72;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_71;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_70;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_63;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_62;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_61;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_54;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_53;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_52;
		
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_81;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_80;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_79;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_78;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_77;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_76;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_75;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_74;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_73;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_72;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_71;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_70;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_63;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_62;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_61;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_54;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_53;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_52;
		
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_81;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_80;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_79;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_78;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_77;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_76;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_75;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_74;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_73;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_72;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_71;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_70;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_63;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_62;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_61;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_54;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_53;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_52;
		
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_81;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_80;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_79;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_78;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_77;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_76;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_75;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_74;
		*(Tensor_64 + Intermediate_3*9 + 8) = Intermediate_73;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_72;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_71;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_70;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_63;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_62;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_61;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_54;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_53;
		*(Tensor_64 + Intermediate_8*9 + 8) = Intermediate_52;
		
	}
	long long end = current_timestamp(); mil += end-start; printf("c module Function_grad: %lld\n", mil);
}

void Function_coupledGrad(int n, const scalar* Tensor_79, const scalar* Tensor_80, const scalar* Tensor_81, const scalar* Tensor_0, const scalar* Tensor_1, const scalar* Tensor_2, const scalar* Tensor_3, const scalar* Tensor_4, const scalar* Tensor_5, const scalar* Tensor_6, const scalar* Tensor_7, const integer* Tensor_8, const integer* Tensor_9, scalar* Tensor_109, scalar* Tensor_113, scalar* Tensor_117) {
	long long start = current_timestamp();
	for (integer i = 0; i < n; i++) {
		integer Intermediate_0 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_1 = *(Tensor_5 + i*3 + 2);
		scalar Intermediate_2 = *(Tensor_0 + i*1 + 0);
		integer Intermediate_3 = *(Tensor_8 + i*1 + 0);
		
		scalar Intermediate_5 = *(Tensor_81 + Intermediate_3*1 + 0);
		scalar Intermediate_6 = *(Tensor_3 + i*1 + 0);
		scalar Intermediate_7 = Intermediate_6*Intermediate_5;
		integer Intermediate_8 = *(Tensor_9 + i*1 + 0);
		scalar Intermediate_9 = *(Tensor_81 + Intermediate_8*1 + 0);
		const scalar Intermediate_10 = -1;
		scalar Intermediate_11 = Intermediate_10*Intermediate_6;
		const scalar Intermediate_12 = 1;
		scalar Intermediate_13 = Intermediate_12+Intermediate_11;
		scalar Intermediate_14 = Intermediate_13*Intermediate_9;
		scalar Intermediate_15 = Intermediate_14+Intermediate_7;
		scalar Intermediate_16 = *(Tensor_1 + i*1 + 0);
		scalar Intermediate_17 = pow(Intermediate_16,Intermediate_10);
		scalar Intermediate_18 = Intermediate_17*Intermediate_15*Intermediate_2*Intermediate_1;
		scalar Intermediate_19 = *(Tensor_5 + i*3 + 1);
		scalar Intermediate_20 = Intermediate_17*Intermediate_15*Intermediate_2*Intermediate_19;
		scalar Intermediate_21 = *(Tensor_5 + i*3 + 0);
		scalar Intermediate_22 = Intermediate_17*Intermediate_15*Intermediate_2*Intermediate_21;
		*(Tensor_117 + Intermediate_3*3 + 2) = Intermediate_22;
		*(Tensor_117 + Intermediate_3*3 + 2) = Intermediate_20;
		*(Tensor_117 + Intermediate_3*3 + 2) = Intermediate_18;
		
		*(Tensor_117 + Intermediate_3*3 + 2) = Intermediate_22;
		*(Tensor_117 + Intermediate_3*3 + 2) = Intermediate_20;
		*(Tensor_117 + Intermediate_3*3 + 2) = Intermediate_18;
		
		*(Tensor_117 + Intermediate_3*3 + 2) = Intermediate_22;
		*(Tensor_117 + Intermediate_3*3 + 2) = Intermediate_20;
		*(Tensor_117 + Intermediate_3*3 + 2) = Intermediate_18;
		
		
		scalar Intermediate_27 = *(Tensor_80 + Intermediate_3*1 + 0);
		scalar Intermediate_28 = Intermediate_6*Intermediate_27;
		scalar Intermediate_29 = *(Tensor_80 + Intermediate_8*1 + 0);
		scalar Intermediate_30 = Intermediate_13*Intermediate_29;
		scalar Intermediate_31 = Intermediate_30+Intermediate_28;
		scalar Intermediate_32 = Intermediate_17*Intermediate_31*Intermediate_2*Intermediate_1;
		scalar Intermediate_33 = Intermediate_17*Intermediate_31*Intermediate_2*Intermediate_19;
		scalar Intermediate_34 = Intermediate_17*Intermediate_31*Intermediate_2*Intermediate_21;
		*(Tensor_113 + Intermediate_3*3 + 2) = Intermediate_34;
		*(Tensor_113 + Intermediate_3*3 + 2) = Intermediate_33;
		*(Tensor_113 + Intermediate_3*3 + 2) = Intermediate_32;
		
		*(Tensor_113 + Intermediate_3*3 + 2) = Intermediate_34;
		*(Tensor_113 + Intermediate_3*3 + 2) = Intermediate_33;
		*(Tensor_113 + Intermediate_3*3 + 2) = Intermediate_32;
		
		*(Tensor_113 + Intermediate_3*3 + 2) = Intermediate_34;
		*(Tensor_113 + Intermediate_3*3 + 2) = Intermediate_33;
		*(Tensor_113 + Intermediate_3*3 + 2) = Intermediate_32;
		
		
		scalar Intermediate_39 = *(Tensor_79 + Intermediate_3*3 + 2);
		scalar Intermediate_40 = Intermediate_6*Intermediate_39;
		scalar Intermediate_41 = *(Tensor_79 + Intermediate_8*3 + 2);
		scalar Intermediate_42 = Intermediate_13*Intermediate_41;
		scalar Intermediate_43 = Intermediate_42+Intermediate_40;
		scalar Intermediate_44 = Intermediate_17*Intermediate_43*Intermediate_2*Intermediate_1;
		scalar Intermediate_45 = Intermediate_17*Intermediate_43*Intermediate_2*Intermediate_19;
		scalar Intermediate_46 = Intermediate_17*Intermediate_43*Intermediate_2*Intermediate_21;
		
		scalar Intermediate_48 = *(Tensor_79 + Intermediate_3*3 + 1);
		scalar Intermediate_49 = Intermediate_6*Intermediate_48;
		scalar Intermediate_50 = *(Tensor_79 + Intermediate_8*3 + 1);
		scalar Intermediate_51 = Intermediate_13*Intermediate_50;
		scalar Intermediate_52 = Intermediate_51+Intermediate_49;
		scalar Intermediate_53 = Intermediate_17*Intermediate_52*Intermediate_2*Intermediate_1;
		scalar Intermediate_54 = Intermediate_17*Intermediate_52*Intermediate_2*Intermediate_19;
		scalar Intermediate_55 = Intermediate_17*Intermediate_52*Intermediate_2*Intermediate_21;
		
		scalar Intermediate_57 = *(Tensor_79 + Intermediate_3*3 + 0);
		scalar Intermediate_58 = Intermediate_6*Intermediate_57;
		scalar Intermediate_59 = *(Tensor_79 + Intermediate_8*3 + 0);
		scalar Intermediate_60 = Intermediate_13*Intermediate_59;
		scalar Intermediate_61 = Intermediate_60+Intermediate_58;
		scalar Intermediate_62 = Intermediate_17*Intermediate_61*Intermediate_2*Intermediate_1;
		scalar Intermediate_63 = Intermediate_17*Intermediate_61*Intermediate_2*Intermediate_19;
		scalar Intermediate_64 = Intermediate_17*Intermediate_61*Intermediate_2*Intermediate_21;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_64;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_63;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_62;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_55;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_54;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_53;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_46;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_45;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_44;
		
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_64;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_63;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_62;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_55;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_54;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_53;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_46;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_45;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_44;
		
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_64;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_63;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_62;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_55;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_54;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_53;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_46;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_45;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_44;
		
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_64;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_63;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_62;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_55;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_54;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_53;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_46;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_45;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_44;
		
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_64;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_63;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_62;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_55;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_54;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_53;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_46;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_45;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_44;
		
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_64;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_63;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_62;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_55;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_54;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_53;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_46;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_45;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_44;
		
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_64;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_63;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_62;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_55;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_54;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_53;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_46;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_45;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_44;
		
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_64;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_63;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_62;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_55;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_54;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_53;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_46;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_45;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_44;
		
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_64;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_63;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_62;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_55;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_54;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_53;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_46;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_45;
		*(Tensor_109 + Intermediate_3*9 + 8) = Intermediate_44;
		
	}
	long long end = current_timestamp(); mil += end-start; printf("c module Function_coupledGrad: %lld\n", mil);
}

void Function_boundaryGrad(int n, const scalar* Tensor_118, const scalar* Tensor_119, const scalar* Tensor_120, const scalar* Tensor_0, const scalar* Tensor_1, const scalar* Tensor_2, const scalar* Tensor_3, const scalar* Tensor_4, const scalar* Tensor_5, const scalar* Tensor_6, const scalar* Tensor_7, const integer* Tensor_8, const integer* Tensor_9, scalar* Tensor_127, scalar* Tensor_131, scalar* Tensor_135) {
	long long start = current_timestamp();
	for (integer i = 0; i < n; i++) {
		integer Intermediate_0 = *(Tensor_8 + i*1 + 0);
		integer Intermediate_1 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_3 = *(Tensor_120 + Intermediate_1*1 + 0);
		scalar Intermediate_4 = *(Tensor_5 + i*3 + 2);
		scalar Intermediate_5 = *(Tensor_0 + i*1 + 0);
		const scalar Intermediate_6 = -1;
		scalar Intermediate_7 = *(Tensor_1 + i*1 + 0);
		scalar Intermediate_8 = pow(Intermediate_7,Intermediate_6);
		scalar Intermediate_9 = Intermediate_8*Intermediate_5*Intermediate_4*Intermediate_3;
		integer Intermediate_10 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_11 = *(Tensor_5 + i*3 + 1);
		scalar Intermediate_12 = Intermediate_8*Intermediate_5*Intermediate_11*Intermediate_3;
		integer Intermediate_13 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_14 = *(Tensor_5 + i*3 + 0);
		scalar Intermediate_15 = Intermediate_8*Intermediate_5*Intermediate_14*Intermediate_3;
		*(Tensor_135 + Intermediate_13*3 + 2) = Intermediate_15;
		*(Tensor_135 + Intermediate_13*3 + 2) = Intermediate_12;
		*(Tensor_135 + Intermediate_13*3 + 2) = Intermediate_9;
		
		*(Tensor_135 + Intermediate_13*3 + 2) = Intermediate_15;
		*(Tensor_135 + Intermediate_13*3 + 2) = Intermediate_12;
		*(Tensor_135 + Intermediate_13*3 + 2) = Intermediate_9;
		
		*(Tensor_135 + Intermediate_13*3 + 2) = Intermediate_15;
		*(Tensor_135 + Intermediate_13*3 + 2) = Intermediate_12;
		*(Tensor_135 + Intermediate_13*3 + 2) = Intermediate_9;
		
		
		scalar Intermediate_20 = *(Tensor_119 + Intermediate_1*1 + 0);
		scalar Intermediate_21 = Intermediate_8*Intermediate_5*Intermediate_4*Intermediate_20;
		scalar Intermediate_22 = Intermediate_8*Intermediate_5*Intermediate_11*Intermediate_20;
		scalar Intermediate_23 = Intermediate_8*Intermediate_5*Intermediate_14*Intermediate_20;
		*(Tensor_131 + Intermediate_13*3 + 2) = Intermediate_23;
		*(Tensor_131 + Intermediate_13*3 + 2) = Intermediate_22;
		*(Tensor_131 + Intermediate_13*3 + 2) = Intermediate_21;
		
		*(Tensor_131 + Intermediate_13*3 + 2) = Intermediate_23;
		*(Tensor_131 + Intermediate_13*3 + 2) = Intermediate_22;
		*(Tensor_131 + Intermediate_13*3 + 2) = Intermediate_21;
		
		*(Tensor_131 + Intermediate_13*3 + 2) = Intermediate_23;
		*(Tensor_131 + Intermediate_13*3 + 2) = Intermediate_22;
		*(Tensor_131 + Intermediate_13*3 + 2) = Intermediate_21;
		
		
		scalar Intermediate_28 = *(Tensor_118 + Intermediate_1*3 + 2);
		scalar Intermediate_29 = Intermediate_8*Intermediate_5*Intermediate_4*Intermediate_28;
		scalar Intermediate_30 = Intermediate_8*Intermediate_5*Intermediate_11*Intermediate_28;
		scalar Intermediate_31 = Intermediate_8*Intermediate_5*Intermediate_14*Intermediate_28;
		
		scalar Intermediate_33 = *(Tensor_118 + Intermediate_1*3 + 1);
		scalar Intermediate_34 = Intermediate_8*Intermediate_5*Intermediate_4*Intermediate_33;
		scalar Intermediate_35 = Intermediate_8*Intermediate_5*Intermediate_11*Intermediate_33;
		scalar Intermediate_36 = Intermediate_8*Intermediate_5*Intermediate_14*Intermediate_33;
		
		scalar Intermediate_38 = *(Tensor_118 + Intermediate_1*3 + 0);
		scalar Intermediate_39 = Intermediate_8*Intermediate_5*Intermediate_4*Intermediate_38;
		scalar Intermediate_40 = Intermediate_8*Intermediate_5*Intermediate_11*Intermediate_38;
		scalar Intermediate_41 = Intermediate_8*Intermediate_5*Intermediate_14*Intermediate_38;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_41;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_40;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_39;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_36;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_35;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_34;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_31;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_30;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_29;
		
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_41;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_40;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_39;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_36;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_35;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_34;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_31;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_30;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_29;
		
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_41;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_40;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_39;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_36;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_35;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_34;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_31;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_30;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_29;
		
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_41;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_40;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_39;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_36;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_35;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_34;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_31;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_30;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_29;
		
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_41;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_40;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_39;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_36;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_35;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_34;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_31;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_30;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_29;
		
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_41;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_40;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_39;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_36;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_35;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_34;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_31;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_30;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_29;
		
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_41;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_40;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_39;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_36;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_35;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_34;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_31;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_30;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_29;
		
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_41;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_40;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_39;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_36;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_35;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_34;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_31;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_30;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_29;
		
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_41;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_40;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_39;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_36;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_35;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_34;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_31;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_30;
		*(Tensor_127 + Intermediate_13*9 + 8) = Intermediate_29;
		
	}
	long long end = current_timestamp(); mil += end-start; printf("c module Function_boundaryGrad: %lld\n", mil);
}

void Function_flux(int n, const scalar* Tensor_136, const scalar* Tensor_137, const scalar* Tensor_138, const scalar* Tensor_139, const scalar* Tensor_140, const scalar* Tensor_141, const scalar* Tensor_0, const scalar* Tensor_1, const scalar* Tensor_2, const scalar* Tensor_3, const scalar* Tensor_4, const scalar* Tensor_5, const scalar* Tensor_6, const scalar* Tensor_7, const integer* Tensor_8, const integer* Tensor_9, scalar* Tensor_529, scalar* Tensor_535, scalar* Tensor_541) {
	long long start = current_timestamp();
	for (integer i = 0; i < n; i++) {
		integer Intermediate_0 = *(Tensor_9 + i*1 + 0);
		scalar Intermediate_1 = *(Tensor_0 + i*1 + 0);
		integer Intermediate_2 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_4 = *(Tensor_137 + Intermediate_2*1 + 0);
		integer Intermediate_5 = *(Tensor_8 + i*1 + 0);
		
		scalar Intermediate_7 = *(Tensor_137 + Intermediate_5*1 + 0);
		integer Intermediate_8 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_10 = *(Tensor_140 + Intermediate_8*3 + 2);
		scalar Intermediate_11 = *(Tensor_7 + i*6 + 5);
		scalar Intermediate_12 = Intermediate_11*Intermediate_10;
		integer Intermediate_13 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_15 = *(Tensor_140 + Intermediate_13*3 + 1);
		scalar Intermediate_16 = *(Tensor_7 + i*6 + 4);
		scalar Intermediate_17 = Intermediate_16*Intermediate_15;
		integer Intermediate_18 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_20 = *(Tensor_140 + Intermediate_18*3 + 0);
		scalar Intermediate_21 = *(Tensor_7 + i*6 + 3);
		scalar Intermediate_22 = Intermediate_21*Intermediate_20;
		integer Intermediate_23 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_24 = *(Tensor_140 + Intermediate_23*3 + 2);
		scalar Intermediate_25 = *(Tensor_7 + i*6 + 2);
		scalar Intermediate_26 = Intermediate_25*Intermediate_24;
		integer Intermediate_27 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_28 = *(Tensor_140 + Intermediate_27*3 + 1);
		scalar Intermediate_29 = *(Tensor_7 + i*6 + 1);
		scalar Intermediate_30 = Intermediate_29*Intermediate_28;
		integer Intermediate_31 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_32 = *(Tensor_140 + Intermediate_31*3 + 0);
		scalar Intermediate_33 = *(Tensor_7 + i*6 + 0);
		scalar Intermediate_34 = Intermediate_33*Intermediate_32;
		scalar Intermediate_35 = *(Tensor_6 + i*2 + 1);
		integer Intermediate_36 = *(Tensor_8 + i*1 + 0);
		
		scalar Intermediate_38 = *(Tensor_137 + Intermediate_36*1 + 0);
		integer Intermediate_39 = *(Tensor_9 + i*1 + 0);
		scalar Intermediate_40 = *(Tensor_137 + Intermediate_39*1 + 0);
		const scalar Intermediate_41 = -1;
		scalar Intermediate_42 = Intermediate_41*Intermediate_40;
		scalar Intermediate_43 = Intermediate_42+Intermediate_38;
		scalar Intermediate_44 = Intermediate_43*Intermediate_35;
		scalar Intermediate_45 = *(Tensor_6 + i*2 + 0);
		const scalar Intermediate_46 = -1;
		scalar Intermediate_47 = Intermediate_46*Intermediate_38;
		scalar Intermediate_48 = Intermediate_47+Intermediate_40;
		scalar Intermediate_49 = Intermediate_48*Intermediate_45;
		const scalar Intermediate_50 = 0.500025;
		scalar Intermediate_51 = Intermediate_50+Intermediate_49+Intermediate_44+Intermediate_34+Intermediate_30+Intermediate_26+Intermediate_22+Intermediate_17+Intermediate_12+Intermediate_38+Intermediate_40;
		const scalar Intermediate_52 = -1;
		scalar Intermediate_53 = *(Tensor_4 + i*1 + 0);
		scalar Intermediate_54 = pow(Intermediate_53,Intermediate_52);
		const scalar Intermediate_55 = 0.5;
		scalar Intermediate_56 = Intermediate_55+Intermediate_49+Intermediate_44+Intermediate_34+Intermediate_30+Intermediate_26+Intermediate_22+Intermediate_17+Intermediate_12+Intermediate_38+Intermediate_40;
		scalar Intermediate_57 = pow(Intermediate_56,Intermediate_52);
		const scalar Intermediate_58 = -1435.0;
		scalar Intermediate_59 = Intermediate_58*Intermediate_57*Intermediate_54*Intermediate_48*Intermediate_51;
		integer Intermediate_60 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_62 = *(Tensor_136 + Intermediate_60*3 + 2);
		integer Intermediate_63 = *(Tensor_8 + i*1 + 0);
		
		scalar Intermediate_65 = *(Tensor_136 + Intermediate_63*3 + 2);
		integer Intermediate_66 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_68 = *(Tensor_139 + Intermediate_66*9 + 8);
		scalar Intermediate_69 = *(Tensor_7 + i*6 + 5);
		scalar Intermediate_70 = Intermediate_69*Intermediate_68;
		integer Intermediate_71 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_73 = *(Tensor_139 + Intermediate_71*9 + 7);
		scalar Intermediate_74 = *(Tensor_7 + i*6 + 4);
		scalar Intermediate_75 = Intermediate_74*Intermediate_73;
		integer Intermediate_76 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_78 = *(Tensor_139 + Intermediate_76*9 + 6);
		scalar Intermediate_79 = *(Tensor_7 + i*6 + 3);
		scalar Intermediate_80 = Intermediate_79*Intermediate_78;
		integer Intermediate_81 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_82 = *(Tensor_139 + Intermediate_81*9 + 8);
		scalar Intermediate_83 = *(Tensor_7 + i*6 + 2);
		scalar Intermediate_84 = Intermediate_83*Intermediate_82;
		integer Intermediate_85 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_86 = *(Tensor_139 + Intermediate_85*9 + 7);
		scalar Intermediate_87 = *(Tensor_7 + i*6 + 1);
		scalar Intermediate_88 = Intermediate_87*Intermediate_86;
		integer Intermediate_89 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_90 = *(Tensor_139 + Intermediate_89*9 + 6);
		scalar Intermediate_91 = *(Tensor_7 + i*6 + 0);
		scalar Intermediate_92 = Intermediate_91*Intermediate_90;
		scalar Intermediate_93 = *(Tensor_6 + i*2 + 1);
		integer Intermediate_94 = *(Tensor_8 + i*1 + 0);
		
		scalar Intermediate_96 = *(Tensor_136 + Intermediate_94*3 + 2);
		integer Intermediate_97 = *(Tensor_9 + i*1 + 0);
		scalar Intermediate_98 = *(Tensor_136 + Intermediate_97*3 + 2);
		scalar Intermediate_99 = Intermediate_52*Intermediate_98;
		scalar Intermediate_100 = Intermediate_99+Intermediate_96;
		scalar Intermediate_101 = Intermediate_100*Intermediate_93;
		scalar Intermediate_102 = *(Tensor_6 + i*2 + 0);
		scalar Intermediate_103 = Intermediate_52*Intermediate_96;
		scalar Intermediate_104 = Intermediate_103+Intermediate_98;
		scalar Intermediate_105 = Intermediate_104*Intermediate_102;
		scalar Intermediate_106 = Intermediate_55+Intermediate_105+Intermediate_101+Intermediate_92+Intermediate_88+Intermediate_84+Intermediate_80+Intermediate_75+Intermediate_70+Intermediate_96+Intermediate_98;
		scalar Intermediate_107 = *(Tensor_5 + i*3 + 2);
		integer Intermediate_108 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_110 = *(Tensor_139 + Intermediate_108*9 + 4);
		integer Intermediate_111 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_112 = *(Tensor_139 + Intermediate_111*9 + 4);
		
		scalar Intermediate_114 = *(Tensor_139 + Intermediate_108*9 + 0);
		scalar Intermediate_115 = *(Tensor_139 + Intermediate_111*9 + 0);
		const scalar Intermediate_116 = 2.16666666666667;
		scalar Intermediate_117 = Intermediate_116+Intermediate_115+Intermediate_114+Intermediate_112+Intermediate_110+Intermediate_82+Intermediate_68;
		scalar Intermediate_118 = Intermediate_52*Intermediate_117*Intermediate_107;
		scalar Intermediate_119 = *(Tensor_5 + i*3 + 1);
		
		scalar Intermediate_121 = *(Tensor_139 + Intermediate_108*9 + 5);
		scalar Intermediate_122 = *(Tensor_139 + Intermediate_111*9 + 5);
		const scalar Intermediate_123 = 1.0;
		scalar Intermediate_124 = Intermediate_123+Intermediate_122+Intermediate_121+Intermediate_86+Intermediate_73;
		scalar Intermediate_125 = Intermediate_124*Intermediate_119;
		scalar Intermediate_126 = *(Tensor_5 + i*3 + 0);
		
		scalar Intermediate_128 = *(Tensor_139 + Intermediate_108*9 + 2);
		scalar Intermediate_129 = *(Tensor_139 + Intermediate_111*9 + 2);
		scalar Intermediate_130 = Intermediate_123+Intermediate_129+Intermediate_128+Intermediate_90+Intermediate_78;
		scalar Intermediate_131 = Intermediate_130*Intermediate_126;
		const scalar Intermediate_132 = 2;
		scalar Intermediate_133 = Intermediate_132*Intermediate_68;
		scalar Intermediate_134 = Intermediate_132*Intermediate_82;
		scalar Intermediate_135 = Intermediate_123+Intermediate_134+Intermediate_133;
		scalar Intermediate_136 = Intermediate_135*Intermediate_107;
		scalar Intermediate_137 = Intermediate_136+Intermediate_131+Intermediate_125+Intermediate_118;
		scalar Intermediate_138 = Intermediate_52*Intermediate_57*Intermediate_137*Intermediate_106*Intermediate_51;
		
		scalar Intermediate_140 = *(Tensor_136 + Intermediate_108*3 + 1);
		
		scalar Intermediate_142 = *(Tensor_136 + Intermediate_111*3 + 1);
		scalar Intermediate_143 = Intermediate_69*Intermediate_121;
		scalar Intermediate_144 = Intermediate_74*Intermediate_110;
		
		scalar Intermediate_146 = *(Tensor_139 + Intermediate_108*9 + 3);
		scalar Intermediate_147 = Intermediate_79*Intermediate_146;
		scalar Intermediate_148 = Intermediate_83*Intermediate_122;
		scalar Intermediate_149 = Intermediate_87*Intermediate_112;
		scalar Intermediate_150 = *(Tensor_139 + Intermediate_111*9 + 3);
		scalar Intermediate_151 = Intermediate_91*Intermediate_150;
		
		scalar Intermediate_153 = *(Tensor_136 + Intermediate_111*3 + 1);
		scalar Intermediate_154 = *(Tensor_136 + Intermediate_108*3 + 1);
		scalar Intermediate_155 = Intermediate_52*Intermediate_154;
		scalar Intermediate_156 = Intermediate_155+Intermediate_153;
		scalar Intermediate_157 = Intermediate_156*Intermediate_93;
		scalar Intermediate_158 = Intermediate_52*Intermediate_153;
		scalar Intermediate_159 = Intermediate_158+Intermediate_154;
		scalar Intermediate_160 = Intermediate_159*Intermediate_102;
		scalar Intermediate_161 = Intermediate_55+Intermediate_160+Intermediate_157+Intermediate_151+Intermediate_149+Intermediate_148+Intermediate_147+Intermediate_144+Intermediate_143+Intermediate_153+Intermediate_154;
		scalar Intermediate_162 = Intermediate_52*Intermediate_117*Intermediate_119;
		scalar Intermediate_163 = Intermediate_124*Intermediate_107;
		
		scalar Intermediate_165 = *(Tensor_139 + Intermediate_108*9 + 1);
		scalar Intermediate_166 = *(Tensor_139 + Intermediate_111*9 + 1);
		scalar Intermediate_167 = Intermediate_123+Intermediate_166+Intermediate_165+Intermediate_150+Intermediate_146;
		scalar Intermediate_168 = Intermediate_167*Intermediate_126;
		scalar Intermediate_169 = Intermediate_132*Intermediate_110;
		scalar Intermediate_170 = Intermediate_132*Intermediate_112;
		scalar Intermediate_171 = Intermediate_123+Intermediate_170+Intermediate_169;
		scalar Intermediate_172 = Intermediate_171*Intermediate_119;
		scalar Intermediate_173 = Intermediate_172+Intermediate_168+Intermediate_163+Intermediate_162;
		scalar Intermediate_174 = Intermediate_52*Intermediate_57*Intermediate_173*Intermediate_161*Intermediate_51;
		
		scalar Intermediate_176 = *(Tensor_136 + Intermediate_108*3 + 0);
		
		scalar Intermediate_178 = *(Tensor_136 + Intermediate_111*3 + 0);
		scalar Intermediate_179 = Intermediate_69*Intermediate_128;
		scalar Intermediate_180 = Intermediate_74*Intermediate_165;
		scalar Intermediate_181 = Intermediate_79*Intermediate_114;
		scalar Intermediate_182 = Intermediate_83*Intermediate_129;
		scalar Intermediate_183 = Intermediate_87*Intermediate_166;
		scalar Intermediate_184 = Intermediate_91*Intermediate_115;
		
		scalar Intermediate_186 = *(Tensor_136 + Intermediate_111*3 + 0);
		scalar Intermediate_187 = *(Tensor_136 + Intermediate_108*3 + 0);
		scalar Intermediate_188 = Intermediate_52*Intermediate_187;
		scalar Intermediate_189 = Intermediate_188+Intermediate_186;
		scalar Intermediate_190 = Intermediate_189*Intermediate_93;
		scalar Intermediate_191 = Intermediate_52*Intermediate_186;
		scalar Intermediate_192 = Intermediate_191+Intermediate_187;
		scalar Intermediate_193 = Intermediate_192*Intermediate_102;
		scalar Intermediate_194 = Intermediate_55+Intermediate_193+Intermediate_190+Intermediate_184+Intermediate_183+Intermediate_182+Intermediate_181+Intermediate_180+Intermediate_179+Intermediate_186+Intermediate_187;
		scalar Intermediate_195 = Intermediate_52*Intermediate_117*Intermediate_126;
		scalar Intermediate_196 = Intermediate_130*Intermediate_107;
		scalar Intermediate_197 = Intermediate_167*Intermediate_119;
		scalar Intermediate_198 = Intermediate_132*Intermediate_114;
		scalar Intermediate_199 = Intermediate_132*Intermediate_115;
		scalar Intermediate_200 = Intermediate_123+Intermediate_199+Intermediate_198;
		scalar Intermediate_201 = Intermediate_200*Intermediate_126;
		scalar Intermediate_202 = Intermediate_201+Intermediate_197+Intermediate_196+Intermediate_195;
		scalar Intermediate_203 = Intermediate_52*Intermediate_57*Intermediate_202*Intermediate_194*Intermediate_51;
		
		scalar Intermediate_205 = *(Tensor_138 + Intermediate_108*1 + 0);
		
		scalar Intermediate_207 = *(Tensor_141 + Intermediate_108*3 + 2);
		scalar Intermediate_208 = Intermediate_69*Intermediate_207;
		
		scalar Intermediate_210 = *(Tensor_141 + Intermediate_108*3 + 1);
		scalar Intermediate_211 = Intermediate_74*Intermediate_210;
		
		scalar Intermediate_213 = *(Tensor_141 + Intermediate_108*3 + 0);
		scalar Intermediate_214 = Intermediate_79*Intermediate_213;
		
		scalar Intermediate_216 = *(Tensor_138 + Intermediate_111*1 + 0);
		
		scalar Intermediate_218 = *(Tensor_138 + Intermediate_108*1 + 0);
		scalar Intermediate_219 = Intermediate_52*Intermediate_218;
		scalar Intermediate_220 = Intermediate_219+Intermediate_216;
		scalar Intermediate_221 = Intermediate_220*Intermediate_93;
		scalar Intermediate_222 = Intermediate_221+Intermediate_214+Intermediate_211+Intermediate_208+Intermediate_218;
		
		scalar Intermediate_224 = *(Tensor_138 + Intermediate_108*1 + 0);
		
		scalar Intermediate_226 = *(Tensor_141 + Intermediate_108*3 + 2);
		scalar Intermediate_227 = Intermediate_69*Intermediate_226;
		
		scalar Intermediate_229 = *(Tensor_141 + Intermediate_108*3 + 1);
		scalar Intermediate_230 = Intermediate_74*Intermediate_229;
		
		scalar Intermediate_232 = *(Tensor_141 + Intermediate_108*3 + 0);
		scalar Intermediate_233 = Intermediate_79*Intermediate_232;
		
		scalar Intermediate_235 = *(Tensor_138 + Intermediate_111*1 + 0);
		
		scalar Intermediate_237 = *(Tensor_138 + Intermediate_108*1 + 0);
		scalar Intermediate_238 = Intermediate_52*Intermediate_237;
		scalar Intermediate_239 = Intermediate_238+Intermediate_235;
		scalar Intermediate_240 = Intermediate_239*Intermediate_93;
		const scalar Intermediate_241 = 1.4;
		scalar Intermediate_242 = Intermediate_241+Intermediate_240+Intermediate_233+Intermediate_230+Intermediate_227+Intermediate_237;
		scalar Intermediate_243 = Intermediate_240+Intermediate_233+Intermediate_230+Intermediate_227+Intermediate_237;
		const scalar Intermediate_244 = 0.4;
		scalar Intermediate_245 = Intermediate_244*Intermediate_69*Intermediate_10;
		scalar Intermediate_246 = Intermediate_244*Intermediate_74*Intermediate_15;
		scalar Intermediate_247 = Intermediate_244*Intermediate_79*Intermediate_20;
		scalar Intermediate_248 = Intermediate_244*Intermediate_43*Intermediate_93;
		scalar Intermediate_249 = Intermediate_244*Intermediate_40;
		const scalar Intermediate_250 = 287.0;
		scalar Intermediate_251 = Intermediate_250+Intermediate_249+Intermediate_248+Intermediate_247+Intermediate_246+Intermediate_245;
		scalar Intermediate_252 = pow(Intermediate_251,Intermediate_52);
		scalar Intermediate_253 = Intermediate_252*Intermediate_243;
		scalar Intermediate_254 = Intermediate_244+Intermediate_253;
		scalar Intermediate_255 = pow(Intermediate_254,Intermediate_52);
		scalar Intermediate_256 = Intermediate_255*Intermediate_242;
		scalar Intermediate_257 = Intermediate_101+Intermediate_80+Intermediate_75+Intermediate_70+Intermediate_98;
		scalar Intermediate_258 = pow(Intermediate_257,Intermediate_132);
		scalar Intermediate_259 = Intermediate_157+Intermediate_147+Intermediate_144+Intermediate_143+Intermediate_154;
		scalar Intermediate_260 = pow(Intermediate_259,Intermediate_132);
		scalar Intermediate_261 = Intermediate_190+Intermediate_181+Intermediate_180+Intermediate_179+Intermediate_187;
		scalar Intermediate_262 = pow(Intermediate_261,Intermediate_132);
		scalar Intermediate_263 = Intermediate_55+Intermediate_262+Intermediate_260+Intermediate_258+Intermediate_256;
		scalar Intermediate_264 = Intermediate_257*Intermediate_107;
		scalar Intermediate_265 = Intermediate_259*Intermediate_119;
		scalar Intermediate_266 = Intermediate_261*Intermediate_126;
		scalar Intermediate_267 = Intermediate_266+Intermediate_265+Intermediate_264;
		scalar Intermediate_268 = Intermediate_252*Intermediate_267*Intermediate_263*Intermediate_243;
		
		scalar Intermediate_270 = *(Tensor_138 + Intermediate_111*1 + 0);
		
		scalar Intermediate_272 = *(Tensor_141 + Intermediate_111*3 + 2);
		scalar Intermediate_273 = Intermediate_83*Intermediate_272;
		
		scalar Intermediate_275 = *(Tensor_141 + Intermediate_111*3 + 1);
		scalar Intermediate_276 = Intermediate_87*Intermediate_275;
		
		scalar Intermediate_278 = *(Tensor_141 + Intermediate_111*3 + 0);
		scalar Intermediate_279 = Intermediate_91*Intermediate_278;
		scalar Intermediate_280 = *(Tensor_138 + Intermediate_111*1 + 0);
		scalar Intermediate_281 = Intermediate_52*Intermediate_280;
		scalar Intermediate_282 = Intermediate_281+Intermediate_237;
		scalar Intermediate_283 = Intermediate_282*Intermediate_102;
		scalar Intermediate_284 = Intermediate_283+Intermediate_279+Intermediate_276+Intermediate_273+Intermediate_280;
		scalar Intermediate_285 = *(Tensor_141 + Intermediate_111*3 + 2);
		scalar Intermediate_286 = Intermediate_83*Intermediate_285;
		scalar Intermediate_287 = *(Tensor_141 + Intermediate_111*3 + 1);
		scalar Intermediate_288 = Intermediate_87*Intermediate_287;
		scalar Intermediate_289 = *(Tensor_141 + Intermediate_111*3 + 0);
		scalar Intermediate_290 = Intermediate_91*Intermediate_289;
		scalar Intermediate_291 = Intermediate_52*Intermediate_280;
		scalar Intermediate_292 = Intermediate_291+Intermediate_237;
		scalar Intermediate_293 = Intermediate_292*Intermediate_102;
		scalar Intermediate_294 = Intermediate_241+Intermediate_293+Intermediate_290+Intermediate_288+Intermediate_286+Intermediate_280;
		scalar Intermediate_295 = Intermediate_293+Intermediate_290+Intermediate_288+Intermediate_286+Intermediate_280;
		scalar Intermediate_296 = Intermediate_244*Intermediate_83*Intermediate_24;
		scalar Intermediate_297 = Intermediate_244*Intermediate_87*Intermediate_28;
		scalar Intermediate_298 = Intermediate_244*Intermediate_91*Intermediate_32;
		scalar Intermediate_299 = Intermediate_244*Intermediate_48*Intermediate_102;
		scalar Intermediate_300 = Intermediate_244*Intermediate_38;
		scalar Intermediate_301 = Intermediate_250+Intermediate_300+Intermediate_299+Intermediate_298+Intermediate_297+Intermediate_296;
		scalar Intermediate_302 = pow(Intermediate_301,Intermediate_52);
		scalar Intermediate_303 = Intermediate_302*Intermediate_295;
		scalar Intermediate_304 = Intermediate_244+Intermediate_303;
		scalar Intermediate_305 = pow(Intermediate_304,Intermediate_52);
		scalar Intermediate_306 = Intermediate_305*Intermediate_294;
		scalar Intermediate_307 = Intermediate_105+Intermediate_92+Intermediate_88+Intermediate_84+Intermediate_96;
		scalar Intermediate_308 = pow(Intermediate_307,Intermediate_132);
		scalar Intermediate_309 = Intermediate_160+Intermediate_151+Intermediate_149+Intermediate_148+Intermediate_153;
		scalar Intermediate_310 = pow(Intermediate_309,Intermediate_132);
		scalar Intermediate_311 = Intermediate_193+Intermediate_184+Intermediate_183+Intermediate_182+Intermediate_186;
		scalar Intermediate_312 = pow(Intermediate_311,Intermediate_132);
		scalar Intermediate_313 = Intermediate_55+Intermediate_312+Intermediate_310+Intermediate_308+Intermediate_306;
		scalar Intermediate_314 = Intermediate_307*Intermediate_107;
		scalar Intermediate_315 = Intermediate_309*Intermediate_119;
		scalar Intermediate_316 = Intermediate_311*Intermediate_126;
		scalar Intermediate_317 = Intermediate_316+Intermediate_315+Intermediate_314;
		scalar Intermediate_318 = Intermediate_302*Intermediate_317*Intermediate_313*Intermediate_295;
		scalar Intermediate_319 = Intermediate_52*Intermediate_302*Intermediate_307*Intermediate_295;
		scalar Intermediate_320 = Intermediate_252*Intermediate_257*Intermediate_243;
		scalar Intermediate_321 = Intermediate_320+Intermediate_319;
		scalar Intermediate_322 = Intermediate_52*Intermediate_321*Intermediate_107;
		scalar Intermediate_323 = Intermediate_52*Intermediate_302*Intermediate_309*Intermediate_295;
		scalar Intermediate_324 = Intermediate_252*Intermediate_259*Intermediate_243;
		scalar Intermediate_325 = Intermediate_324+Intermediate_323;
		scalar Intermediate_326 = Intermediate_52*Intermediate_325*Intermediate_119;
		scalar Intermediate_327 = Intermediate_52*Intermediate_302*Intermediate_311*Intermediate_295;
		scalar Intermediate_328 = Intermediate_252*Intermediate_261*Intermediate_243;
		scalar Intermediate_329 = Intermediate_328+Intermediate_327;
		scalar Intermediate_330 = Intermediate_52*Intermediate_329*Intermediate_126;
		const scalar Intermediate_331 = 0.5;
		scalar Intermediate_332 = pow(Intermediate_253,Intermediate_331);
		scalar Intermediate_333 = Intermediate_332*Intermediate_257;
		scalar Intermediate_334 = pow(Intermediate_303,Intermediate_331);
		scalar Intermediate_335 = Intermediate_334*Intermediate_307;
		scalar Intermediate_336 = Intermediate_335+Intermediate_333;
		scalar Intermediate_337 = Intermediate_334+Intermediate_332;
		scalar Intermediate_338 = pow(Intermediate_337,Intermediate_52);
		scalar Intermediate_339 = Intermediate_338*Intermediate_336*Intermediate_107;
		scalar Intermediate_340 = Intermediate_332*Intermediate_259;
		scalar Intermediate_341 = Intermediate_334*Intermediate_309;
		scalar Intermediate_342 = Intermediate_341+Intermediate_340;
		scalar Intermediate_343 = Intermediate_338*Intermediate_342*Intermediate_119;
		scalar Intermediate_344 = Intermediate_332*Intermediate_261;
		scalar Intermediate_345 = Intermediate_334*Intermediate_311;
		scalar Intermediate_346 = Intermediate_345+Intermediate_344;
		scalar Intermediate_347 = Intermediate_338*Intermediate_346*Intermediate_126;
		scalar Intermediate_348 = Intermediate_347+Intermediate_343+Intermediate_339;
		scalar Intermediate_349 = Intermediate_52*Intermediate_302*Intermediate_295;
		scalar Intermediate_350 = Intermediate_253+Intermediate_349;
		scalar Intermediate_351 = Intermediate_350*Intermediate_348;
		scalar Intermediate_352 = Intermediate_351+Intermediate_330+Intermediate_326+Intermediate_322;
		
		
		scalar Intermediate_355 = pow(Intermediate_336,Intermediate_132);
		const scalar Intermediate_356 = -2;
		scalar Intermediate_357 = pow(Intermediate_337,Intermediate_356);
		scalar Intermediate_358 = Intermediate_52*Intermediate_357*Intermediate_355;
		scalar Intermediate_359 = pow(Intermediate_342,Intermediate_132);
		scalar Intermediate_360 = Intermediate_52*Intermediate_357*Intermediate_359;
		scalar Intermediate_361 = pow(Intermediate_346,Intermediate_132);
		scalar Intermediate_362 = Intermediate_52*Intermediate_357*Intermediate_361;
		scalar Intermediate_363 = Intermediate_332*Intermediate_263;
		scalar Intermediate_364 = Intermediate_334*Intermediate_313;
		scalar Intermediate_365 = Intermediate_364+Intermediate_363;
		scalar Intermediate_366 = Intermediate_338*Intermediate_365;
		const scalar Intermediate_367 = -0.1;
		scalar Intermediate_368 = Intermediate_367+Intermediate_366+Intermediate_362+Intermediate_360+Intermediate_358;
		scalar Intermediate_369 = pow(Intermediate_368,Intermediate_331);
		scalar Intermediate_370 = Intermediate_369+Intermediate_347+Intermediate_343+Intermediate_339;
		
		const scalar Intermediate_372 = 0;
		int Intermediate_373 = Intermediate_370 < Intermediate_372;
		scalar Intermediate_374 = Intermediate_52*Intermediate_338*Intermediate_336*Intermediate_107;
		scalar Intermediate_375 = Intermediate_52*Intermediate_338*Intermediate_342*Intermediate_119;
		scalar Intermediate_376 = Intermediate_52*Intermediate_338*Intermediate_346*Intermediate_126;
		scalar Intermediate_377 = Intermediate_52*Intermediate_369;
		scalar Intermediate_378 = Intermediate_377+Intermediate_376+Intermediate_375+Intermediate_374;
		
		
                scalar Intermediate_380;
                if (Intermediate_373) 
                    Intermediate_380 = Intermediate_378;
                else 
                    Intermediate_380 = Intermediate_370;
                
		
		scalar Intermediate_382 = Intermediate_52*Intermediate_257*Intermediate_107;
		scalar Intermediate_383 = Intermediate_52*Intermediate_259*Intermediate_119;
		scalar Intermediate_384 = Intermediate_52*Intermediate_261*Intermediate_126;
		scalar Intermediate_385 = Intermediate_316+Intermediate_315+Intermediate_314+Intermediate_384+Intermediate_383+Intermediate_382;
		
		int Intermediate_387 = Intermediate_385 < Intermediate_372;
		scalar Intermediate_388 = Intermediate_52*Intermediate_307*Intermediate_107;
		scalar Intermediate_389 = Intermediate_52*Intermediate_309*Intermediate_119;
		scalar Intermediate_390 = Intermediate_52*Intermediate_311*Intermediate_126;
		scalar Intermediate_391 = Intermediate_266+Intermediate_265+Intermediate_264+Intermediate_390+Intermediate_389+Intermediate_388;
		
		
                scalar Intermediate_393;
                if (Intermediate_387) 
                    Intermediate_393 = Intermediate_391;
                else 
                    Intermediate_393 = Intermediate_385;
                
		const scalar Intermediate_394 = 2.0;
		scalar Intermediate_395 = Intermediate_394*Intermediate_393;
		scalar Intermediate_396 = pow(Intermediate_243,Intermediate_52);
		scalar Intermediate_397 = Intermediate_396*Intermediate_242*Intermediate_251;
		scalar Intermediate_398 = pow(Intermediate_397,Intermediate_331);
		scalar Intermediate_399 = Intermediate_52*Intermediate_398;
		scalar Intermediate_400 = pow(Intermediate_295,Intermediate_52);
		scalar Intermediate_401 = Intermediate_400*Intermediate_294*Intermediate_301;
		scalar Intermediate_402 = pow(Intermediate_401,Intermediate_331);
		scalar Intermediate_403 = Intermediate_402+Intermediate_399;
		
		int Intermediate_405 = Intermediate_403 < Intermediate_372;
		scalar Intermediate_406 = Intermediate_52*Intermediate_402;
		scalar Intermediate_407 = Intermediate_398+Intermediate_406;
		
		
                scalar Intermediate_409;
                if (Intermediate_405) 
                    Intermediate_409 = Intermediate_407;
                else 
                    Intermediate_409 = Intermediate_403;
                
		scalar Intermediate_410 = Intermediate_394*Intermediate_409;
		scalar Intermediate_411 = Intermediate_394+Intermediate_410+Intermediate_395;
		int Intermediate_412 = Intermediate_380 < Intermediate_411;
		const scalar Intermediate_413 = 0.25;
		scalar Intermediate_414 = Intermediate_413+Intermediate_380;
		scalar Intermediate_415 = Intermediate_123+Intermediate_409+Intermediate_393;
		scalar Intermediate_416 = pow(Intermediate_415,Intermediate_52);
		scalar Intermediate_417 = Intermediate_416*Intermediate_414*Intermediate_380;
		scalar Intermediate_418 = Intermediate_123+Intermediate_417+Intermediate_409+Intermediate_393;
		
		
                scalar Intermediate_420;
                if (Intermediate_412) 
                    Intermediate_420 = Intermediate_418;
                else 
                    Intermediate_420 = Intermediate_380;
                
		scalar Intermediate_421 = Intermediate_377+Intermediate_347+Intermediate_343+Intermediate_339;
		
		int Intermediate_423 = Intermediate_421 < Intermediate_372;
		scalar Intermediate_424 = Intermediate_369+Intermediate_376+Intermediate_375+Intermediate_374;
		
		
                scalar Intermediate_426;
                if (Intermediate_423) 
                    Intermediate_426 = Intermediate_424;
                else 
                    Intermediate_426 = Intermediate_421;
                
		
		int Intermediate_428 = Intermediate_426 < Intermediate_411;
		scalar Intermediate_429 = Intermediate_413+Intermediate_426;
		scalar Intermediate_430 = Intermediate_416*Intermediate_429*Intermediate_426;
		scalar Intermediate_431 = Intermediate_123+Intermediate_430+Intermediate_409+Intermediate_393;
		
		
                scalar Intermediate_433;
                if (Intermediate_428) 
                    Intermediate_433 = Intermediate_431;
                else 
                    Intermediate_433 = Intermediate_426;
                
		scalar Intermediate_434 = Intermediate_52*Intermediate_433;
		scalar Intermediate_435 = Intermediate_55+Intermediate_434+Intermediate_420;
		scalar Intermediate_436 = -0.5;
		scalar Intermediate_437 = pow(Intermediate_368,Intermediate_436);
		scalar Intermediate_438 = Intermediate_52*Intermediate_437*Intermediate_435*Intermediate_352;
		scalar Intermediate_439 = Intermediate_52*Intermediate_302*Intermediate_313*Intermediate_295;
		scalar Intermediate_440 = Intermediate_52*Intermediate_338*Intermediate_336*Intermediate_321;
		scalar Intermediate_441 = Intermediate_52*Intermediate_338*Intermediate_342*Intermediate_325;
		scalar Intermediate_442 = Intermediate_52*Intermediate_338*Intermediate_346*Intermediate_329;
		scalar Intermediate_443 = Intermediate_252*Intermediate_263*Intermediate_243;
		scalar Intermediate_444 = Intermediate_52*Intermediate_69*Intermediate_226;
		scalar Intermediate_445 = Intermediate_52*Intermediate_74*Intermediate_229;
		scalar Intermediate_446 = Intermediate_52*Intermediate_79*Intermediate_232;
		scalar Intermediate_447 = Intermediate_52*Intermediate_239*Intermediate_93;
		scalar Intermediate_448 = Intermediate_357*Intermediate_355;
		scalar Intermediate_449 = Intermediate_357*Intermediate_359;
		scalar Intermediate_450 = Intermediate_357*Intermediate_361;
		scalar Intermediate_451 = Intermediate_55+Intermediate_450+Intermediate_449+Intermediate_448;
		scalar Intermediate_452 = Intermediate_350*Intermediate_451;
		scalar Intermediate_453 = Intermediate_244+Intermediate_238+Intermediate_293+Intermediate_452+Intermediate_290+Intermediate_288+Intermediate_286+Intermediate_447+Intermediate_446+Intermediate_445+Intermediate_444+Intermediate_443+Intermediate_442+Intermediate_441+Intermediate_440+Intermediate_439+Intermediate_280;
		
		int Intermediate_455 = Intermediate_348 < Intermediate_372;
		scalar Intermediate_456 = Intermediate_376+Intermediate_375+Intermediate_374;
		
		
                scalar Intermediate_458;
                if (Intermediate_455) 
                    Intermediate_458 = Intermediate_456;
                else 
                    Intermediate_458 = Intermediate_348;
                
		
		int Intermediate_460 = Intermediate_458 < Intermediate_411;
		scalar Intermediate_461 = Intermediate_413+Intermediate_458;
		scalar Intermediate_462 = Intermediate_416*Intermediate_461*Intermediate_458;
		scalar Intermediate_463 = Intermediate_123+Intermediate_462+Intermediate_409+Intermediate_393;
		
		
                scalar Intermediate_465;
                if (Intermediate_460) 
                    Intermediate_465 = Intermediate_463;
                else 
                    Intermediate_465 = Intermediate_458;
                
		scalar Intermediate_466 = Intermediate_52*Intermediate_465;
		scalar Intermediate_467 = Intermediate_55+Intermediate_466+Intermediate_433+Intermediate_420;
		scalar Intermediate_468 = pow(Intermediate_368,Intermediate_52);
		scalar Intermediate_469 = Intermediate_468*Intermediate_467*Intermediate_453;
		scalar Intermediate_470 = Intermediate_469+Intermediate_438;
		scalar Intermediate_471 = Intermediate_52*Intermediate_338*Intermediate_365*Intermediate_470;
		scalar Intermediate_472 = Intermediate_238+Intermediate_293+Intermediate_290+Intermediate_288+Intermediate_286+Intermediate_447+Intermediate_446+Intermediate_445+Intermediate_444+Intermediate_443+Intermediate_439+Intermediate_280;
		scalar Intermediate_473 = Intermediate_52*Intermediate_472*Intermediate_465;
		scalar Intermediate_474 = Intermediate_52*Intermediate_437*Intermediate_435*Intermediate_453;
		scalar Intermediate_475 = Intermediate_467*Intermediate_352;
		scalar Intermediate_476 = Intermediate_475+Intermediate_474;
		scalar Intermediate_477 = Intermediate_476*Intermediate_348;
		scalar Intermediate_478 = Intermediate_477+Intermediate_473+Intermediate_471+Intermediate_318+Intermediate_268+Intermediate_203+Intermediate_174+Intermediate_138+Intermediate_59;
		scalar Intermediate_479 = *(Tensor_2 + i*1 + 0);
		scalar Intermediate_480 = pow(Intermediate_479,Intermediate_52);
		scalar Intermediate_481 = Intermediate_52*Intermediate_480*Intermediate_478*Intermediate_1;
		scalar Intermediate_482 = *(Tensor_1 + i*1 + 0);
		scalar Intermediate_483 = pow(Intermediate_482,Intermediate_52);
		scalar Intermediate_484 = Intermediate_483*Intermediate_478*Intermediate_1;
		*(Tensor_541 + Intermediate_111*1 + 0) = Intermediate_484;
		*(Tensor_541 + Intermediate_108*1 + 0) = Intermediate_481;
		
		scalar Intermediate_486 = Intermediate_252*Intermediate_267*Intermediate_257*Intermediate_243;
		scalar Intermediate_487 = Intermediate_302*Intermediate_317*Intermediate_307*Intermediate_295;
		scalar Intermediate_488 = Intermediate_52*Intermediate_57*Intermediate_137*Intermediate_51;
		scalar Intermediate_489 = Intermediate_52*Intermediate_338*Intermediate_336*Intermediate_470;
		scalar Intermediate_490 = Intermediate_52*Intermediate_321*Intermediate_465;
		scalar Intermediate_491 = Intermediate_293+Intermediate_240+Intermediate_290+Intermediate_288+Intermediate_286+Intermediate_233+Intermediate_230+Intermediate_227+Intermediate_280+Intermediate_237;
		scalar Intermediate_492 = Intermediate_491*Intermediate_107;
		scalar Intermediate_493 = Intermediate_476*Intermediate_107;
		scalar Intermediate_494 = Intermediate_493+Intermediate_492+Intermediate_490+Intermediate_489+Intermediate_488+Intermediate_487+Intermediate_486;
		scalar Intermediate_495 = Intermediate_52*Intermediate_480*Intermediate_494*Intermediate_1;
		scalar Intermediate_496 = Intermediate_252*Intermediate_267*Intermediate_259*Intermediate_243;
		scalar Intermediate_497 = Intermediate_302*Intermediate_317*Intermediate_309*Intermediate_295;
		scalar Intermediate_498 = Intermediate_52*Intermediate_57*Intermediate_173*Intermediate_51;
		scalar Intermediate_499 = Intermediate_52*Intermediate_338*Intermediate_342*Intermediate_470;
		scalar Intermediate_500 = Intermediate_52*Intermediate_325*Intermediate_465;
		scalar Intermediate_501 = Intermediate_491*Intermediate_119;
		scalar Intermediate_502 = Intermediate_476*Intermediate_119;
		scalar Intermediate_503 = Intermediate_502+Intermediate_501+Intermediate_500+Intermediate_499+Intermediate_498+Intermediate_497+Intermediate_496;
		scalar Intermediate_504 = Intermediate_52*Intermediate_480*Intermediate_503*Intermediate_1;
		scalar Intermediate_505 = Intermediate_252*Intermediate_267*Intermediate_261*Intermediate_243;
		scalar Intermediate_506 = Intermediate_302*Intermediate_317*Intermediate_311*Intermediate_295;
		scalar Intermediate_507 = Intermediate_52*Intermediate_57*Intermediate_202*Intermediate_51;
		scalar Intermediate_508 = Intermediate_52*Intermediate_338*Intermediate_346*Intermediate_470;
		scalar Intermediate_509 = Intermediate_52*Intermediate_329*Intermediate_465;
		scalar Intermediate_510 = Intermediate_491*Intermediate_126;
		scalar Intermediate_511 = Intermediate_476*Intermediate_126;
		scalar Intermediate_512 = Intermediate_511+Intermediate_510+Intermediate_509+Intermediate_508+Intermediate_507+Intermediate_506+Intermediate_505;
		scalar Intermediate_513 = Intermediate_52*Intermediate_480*Intermediate_512*Intermediate_1;
		scalar Intermediate_514 = Intermediate_483*Intermediate_494*Intermediate_1;
		scalar Intermediate_515 = Intermediate_483*Intermediate_503*Intermediate_1;
		scalar Intermediate_516 = Intermediate_483*Intermediate_512*Intermediate_1;
		*(Tensor_535 + Intermediate_111*3 + 2) = Intermediate_516;
		*(Tensor_535 + Intermediate_111*3 + 2) = Intermediate_515;
		*(Tensor_535 + Intermediate_111*3 + 2) = Intermediate_514;
		*(Tensor_535 + Intermediate_108*3 + 2) = Intermediate_513;
		*(Tensor_535 + Intermediate_108*3 + 2) = Intermediate_504;
		*(Tensor_535 + Intermediate_108*3 + 2) = Intermediate_495;
		
		*(Tensor_535 + Intermediate_111*3 + 2) = Intermediate_516;
		*(Tensor_535 + Intermediate_111*3 + 2) = Intermediate_515;
		*(Tensor_535 + Intermediate_111*3 + 2) = Intermediate_514;
		*(Tensor_535 + Intermediate_108*3 + 2) = Intermediate_513;
		*(Tensor_535 + Intermediate_108*3 + 2) = Intermediate_504;
		*(Tensor_535 + Intermediate_108*3 + 2) = Intermediate_495;
		
		*(Tensor_535 + Intermediate_111*3 + 2) = Intermediate_516;
		*(Tensor_535 + Intermediate_111*3 + 2) = Intermediate_515;
		*(Tensor_535 + Intermediate_111*3 + 2) = Intermediate_514;
		*(Tensor_535 + Intermediate_108*3 + 2) = Intermediate_513;
		*(Tensor_535 + Intermediate_108*3 + 2) = Intermediate_504;
		*(Tensor_535 + Intermediate_108*3 + 2) = Intermediate_495;
		
		scalar Intermediate_520 = Intermediate_52*Intermediate_468*Intermediate_467*Intermediate_453;
		scalar Intermediate_521 = Intermediate_252*Intermediate_267*Intermediate_243;
		scalar Intermediate_522 = Intermediate_302*Intermediate_317*Intermediate_295;
		scalar Intermediate_523 = Intermediate_437*Intermediate_435*Intermediate_352;
		scalar Intermediate_524 = Intermediate_52*Intermediate_350*Intermediate_465;
		scalar Intermediate_525 = Intermediate_524+Intermediate_523+Intermediate_522+Intermediate_521+Intermediate_520;
		scalar Intermediate_526 = Intermediate_52*Intermediate_480*Intermediate_525*Intermediate_1;
		scalar Intermediate_527 = Intermediate_483*Intermediate_525*Intermediate_1;
		*(Tensor_529 + Intermediate_111*1 + 0) = Intermediate_527;
		*(Tensor_529 + Intermediate_108*1 + 0) = Intermediate_526;
		
	}
	long long end = current_timestamp(); mil += end-start; printf("c module Function_flux: %lld\n", mil);
}

void Function_characteristicFlux(int n, const scalar* Tensor_542, const scalar* Tensor_543, const scalar* Tensor_544, const scalar* Tensor_545, const scalar* Tensor_546, const scalar* Tensor_547, const scalar* Tensor_0, const scalar* Tensor_1, const scalar* Tensor_2, const scalar* Tensor_3, const scalar* Tensor_4, const scalar* Tensor_5, const scalar* Tensor_6, const scalar* Tensor_7, const integer* Tensor_8, const integer* Tensor_9, scalar* Tensor_932, scalar* Tensor_935, scalar* Tensor_938) {
	long long start = current_timestamp();
	for (integer i = 0; i < n; i++) {
		integer Intermediate_0 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_1 = *(Tensor_0 + i*1 + 0);
		integer Intermediate_2 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_4 = *(Tensor_543 + Intermediate_2*1 + 0);
		integer Intermediate_5 = *(Tensor_8 + i*1 + 0);
		
		scalar Intermediate_7 = *(Tensor_543 + Intermediate_5*1 + 0);
		integer Intermediate_8 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_10 = *(Tensor_546 + Intermediate_8*3 + 2);
		scalar Intermediate_11 = *(Tensor_7 + i*6 + 5);
		scalar Intermediate_12 = Intermediate_11*Intermediate_10;
		integer Intermediate_13 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_15 = *(Tensor_546 + Intermediate_13*3 + 1);
		scalar Intermediate_16 = *(Tensor_7 + i*6 + 4);
		scalar Intermediate_17 = Intermediate_16*Intermediate_15;
		integer Intermediate_18 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_20 = *(Tensor_546 + Intermediate_18*3 + 0);
		scalar Intermediate_21 = *(Tensor_7 + i*6 + 3);
		scalar Intermediate_22 = Intermediate_21*Intermediate_20;
		integer Intermediate_23 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_24 = *(Tensor_546 + Intermediate_23*3 + 2);
		scalar Intermediate_25 = *(Tensor_7 + i*6 + 2);
		scalar Intermediate_26 = Intermediate_25*Intermediate_24;
		integer Intermediate_27 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_28 = *(Tensor_546 + Intermediate_27*3 + 1);
		scalar Intermediate_29 = *(Tensor_7 + i*6 + 1);
		scalar Intermediate_30 = Intermediate_29*Intermediate_28;
		integer Intermediate_31 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_32 = *(Tensor_546 + Intermediate_31*3 + 0);
		scalar Intermediate_33 = *(Tensor_7 + i*6 + 0);
		scalar Intermediate_34 = Intermediate_33*Intermediate_32;
		scalar Intermediate_35 = *(Tensor_6 + i*2 + 1);
		integer Intermediate_36 = *(Tensor_8 + i*1 + 0);
		
		scalar Intermediate_38 = *(Tensor_543 + Intermediate_36*1 + 0);
		integer Intermediate_39 = *(Tensor_9 + i*1 + 0);
		scalar Intermediate_40 = *(Tensor_543 + Intermediate_39*1 + 0);
		const scalar Intermediate_41 = -1;
		scalar Intermediate_42 = Intermediate_41*Intermediate_40;
		scalar Intermediate_43 = Intermediate_42+Intermediate_38;
		scalar Intermediate_44 = Intermediate_43*Intermediate_35;
		scalar Intermediate_45 = *(Tensor_6 + i*2 + 0);
		const scalar Intermediate_46 = -1;
		scalar Intermediate_47 = Intermediate_46*Intermediate_38;
		scalar Intermediate_48 = Intermediate_47+Intermediate_40;
		scalar Intermediate_49 = Intermediate_48*Intermediate_45;
		const scalar Intermediate_50 = 0.500025;
		scalar Intermediate_51 = Intermediate_50+Intermediate_49+Intermediate_44+Intermediate_34+Intermediate_30+Intermediate_26+Intermediate_22+Intermediate_17+Intermediate_12+Intermediate_38+Intermediate_40;
		const scalar Intermediate_52 = -1;
		scalar Intermediate_53 = *(Tensor_4 + i*1 + 0);
		scalar Intermediate_54 = pow(Intermediate_53,Intermediate_52);
		const scalar Intermediate_55 = 0.5;
		scalar Intermediate_56 = Intermediate_55+Intermediate_49+Intermediate_44+Intermediate_34+Intermediate_30+Intermediate_26+Intermediate_22+Intermediate_17+Intermediate_12+Intermediate_38+Intermediate_40;
		scalar Intermediate_57 = pow(Intermediate_56,Intermediate_52);
		const scalar Intermediate_58 = -1435.0;
		scalar Intermediate_59 = Intermediate_58*Intermediate_57*Intermediate_54*Intermediate_48*Intermediate_51;
		integer Intermediate_60 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_62 = *(Tensor_542 + Intermediate_60*3 + 2);
		integer Intermediate_63 = *(Tensor_8 + i*1 + 0);
		
		scalar Intermediate_65 = *(Tensor_542 + Intermediate_63*3 + 2);
		integer Intermediate_66 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_68 = *(Tensor_545 + Intermediate_66*9 + 8);
		scalar Intermediate_69 = *(Tensor_7 + i*6 + 5);
		scalar Intermediate_70 = Intermediate_69*Intermediate_68;
		integer Intermediate_71 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_73 = *(Tensor_545 + Intermediate_71*9 + 7);
		scalar Intermediate_74 = *(Tensor_7 + i*6 + 4);
		scalar Intermediate_75 = Intermediate_74*Intermediate_73;
		integer Intermediate_76 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_78 = *(Tensor_545 + Intermediate_76*9 + 6);
		scalar Intermediate_79 = *(Tensor_7 + i*6 + 3);
		scalar Intermediate_80 = Intermediate_79*Intermediate_78;
		integer Intermediate_81 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_82 = *(Tensor_545 + Intermediate_81*9 + 8);
		scalar Intermediate_83 = *(Tensor_7 + i*6 + 2);
		scalar Intermediate_84 = Intermediate_83*Intermediate_82;
		integer Intermediate_85 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_86 = *(Tensor_545 + Intermediate_85*9 + 7);
		scalar Intermediate_87 = *(Tensor_7 + i*6 + 1);
		scalar Intermediate_88 = Intermediate_87*Intermediate_86;
		integer Intermediate_89 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_90 = *(Tensor_545 + Intermediate_89*9 + 6);
		scalar Intermediate_91 = *(Tensor_7 + i*6 + 0);
		scalar Intermediate_92 = Intermediate_91*Intermediate_90;
		scalar Intermediate_93 = *(Tensor_6 + i*2 + 1);
		integer Intermediate_94 = *(Tensor_8 + i*1 + 0);
		
		scalar Intermediate_96 = *(Tensor_542 + Intermediate_94*3 + 2);
		integer Intermediate_97 = *(Tensor_9 + i*1 + 0);
		scalar Intermediate_98 = *(Tensor_542 + Intermediate_97*3 + 2);
		scalar Intermediate_99 = Intermediate_52*Intermediate_98;
		scalar Intermediate_100 = Intermediate_99+Intermediate_96;
		scalar Intermediate_101 = Intermediate_100*Intermediate_93;
		scalar Intermediate_102 = *(Tensor_6 + i*2 + 0);
		scalar Intermediate_103 = Intermediate_52*Intermediate_96;
		scalar Intermediate_104 = Intermediate_103+Intermediate_98;
		scalar Intermediate_105 = Intermediate_104*Intermediate_102;
		scalar Intermediate_106 = Intermediate_55+Intermediate_105+Intermediate_101+Intermediate_92+Intermediate_88+Intermediate_84+Intermediate_80+Intermediate_75+Intermediate_70+Intermediate_96+Intermediate_98;
		scalar Intermediate_107 = *(Tensor_5 + i*3 + 2);
		integer Intermediate_108 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_110 = *(Tensor_545 + Intermediate_108*9 + 4);
		integer Intermediate_111 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_112 = *(Tensor_545 + Intermediate_111*9 + 4);
		
		scalar Intermediate_114 = *(Tensor_545 + Intermediate_108*9 + 0);
		scalar Intermediate_115 = *(Tensor_545 + Intermediate_111*9 + 0);
		const scalar Intermediate_116 = 2.16666666666667;
		scalar Intermediate_117 = Intermediate_116+Intermediate_115+Intermediate_114+Intermediate_112+Intermediate_110+Intermediate_82+Intermediate_68;
		scalar Intermediate_118 = Intermediate_52*Intermediate_117*Intermediate_107;
		scalar Intermediate_119 = *(Tensor_5 + i*3 + 1);
		
		scalar Intermediate_121 = *(Tensor_545 + Intermediate_108*9 + 5);
		scalar Intermediate_122 = *(Tensor_545 + Intermediate_111*9 + 5);
		const scalar Intermediate_123 = 1.0;
		scalar Intermediate_124 = Intermediate_123+Intermediate_122+Intermediate_121+Intermediate_86+Intermediate_73;
		scalar Intermediate_125 = Intermediate_124*Intermediate_119;
		scalar Intermediate_126 = *(Tensor_5 + i*3 + 0);
		
		scalar Intermediate_128 = *(Tensor_545 + Intermediate_108*9 + 2);
		scalar Intermediate_129 = *(Tensor_545 + Intermediate_111*9 + 2);
		scalar Intermediate_130 = Intermediate_123+Intermediate_129+Intermediate_128+Intermediate_90+Intermediate_78;
		scalar Intermediate_131 = Intermediate_130*Intermediate_126;
		const scalar Intermediate_132 = 2;
		scalar Intermediate_133 = Intermediate_132*Intermediate_68;
		scalar Intermediate_134 = Intermediate_132*Intermediate_82;
		scalar Intermediate_135 = Intermediate_123+Intermediate_134+Intermediate_133;
		scalar Intermediate_136 = Intermediate_135*Intermediate_107;
		scalar Intermediate_137 = Intermediate_136+Intermediate_131+Intermediate_125+Intermediate_118;
		scalar Intermediate_138 = Intermediate_52*Intermediate_57*Intermediate_137*Intermediate_106*Intermediate_51;
		
		scalar Intermediate_140 = *(Tensor_542 + Intermediate_108*3 + 1);
		
		scalar Intermediate_142 = *(Tensor_542 + Intermediate_111*3 + 1);
		scalar Intermediate_143 = Intermediate_69*Intermediate_121;
		scalar Intermediate_144 = Intermediate_74*Intermediate_110;
		
		scalar Intermediate_146 = *(Tensor_545 + Intermediate_108*9 + 3);
		scalar Intermediate_147 = Intermediate_79*Intermediate_146;
		scalar Intermediate_148 = Intermediate_83*Intermediate_122;
		scalar Intermediate_149 = Intermediate_87*Intermediate_112;
		scalar Intermediate_150 = *(Tensor_545 + Intermediate_111*9 + 3);
		scalar Intermediate_151 = Intermediate_91*Intermediate_150;
		
		scalar Intermediate_153 = *(Tensor_542 + Intermediate_111*3 + 1);
		scalar Intermediate_154 = *(Tensor_542 + Intermediate_108*3 + 1);
		scalar Intermediate_155 = Intermediate_52*Intermediate_154;
		scalar Intermediate_156 = Intermediate_155+Intermediate_153;
		scalar Intermediate_157 = Intermediate_156*Intermediate_93;
		scalar Intermediate_158 = Intermediate_52*Intermediate_153;
		scalar Intermediate_159 = Intermediate_158+Intermediate_154;
		scalar Intermediate_160 = Intermediate_159*Intermediate_102;
		scalar Intermediate_161 = Intermediate_55+Intermediate_160+Intermediate_157+Intermediate_151+Intermediate_149+Intermediate_148+Intermediate_147+Intermediate_144+Intermediate_143+Intermediate_153+Intermediate_154;
		scalar Intermediate_162 = Intermediate_52*Intermediate_117*Intermediate_119;
		scalar Intermediate_163 = Intermediate_124*Intermediate_107;
		
		scalar Intermediate_165 = *(Tensor_545 + Intermediate_108*9 + 1);
		scalar Intermediate_166 = *(Tensor_545 + Intermediate_111*9 + 1);
		scalar Intermediate_167 = Intermediate_123+Intermediate_166+Intermediate_165+Intermediate_150+Intermediate_146;
		scalar Intermediate_168 = Intermediate_167*Intermediate_126;
		scalar Intermediate_169 = Intermediate_132*Intermediate_110;
		scalar Intermediate_170 = Intermediate_132*Intermediate_112;
		scalar Intermediate_171 = Intermediate_123+Intermediate_170+Intermediate_169;
		scalar Intermediate_172 = Intermediate_171*Intermediate_119;
		scalar Intermediate_173 = Intermediate_172+Intermediate_168+Intermediate_163+Intermediate_162;
		scalar Intermediate_174 = Intermediate_52*Intermediate_57*Intermediate_173*Intermediate_161*Intermediate_51;
		
		scalar Intermediate_176 = *(Tensor_542 + Intermediate_108*3 + 0);
		
		scalar Intermediate_178 = *(Tensor_542 + Intermediate_111*3 + 0);
		scalar Intermediate_179 = Intermediate_69*Intermediate_128;
		scalar Intermediate_180 = Intermediate_74*Intermediate_165;
		scalar Intermediate_181 = Intermediate_79*Intermediate_114;
		scalar Intermediate_182 = Intermediate_83*Intermediate_129;
		scalar Intermediate_183 = Intermediate_87*Intermediate_166;
		scalar Intermediate_184 = Intermediate_91*Intermediate_115;
		
		scalar Intermediate_186 = *(Tensor_542 + Intermediate_111*3 + 0);
		scalar Intermediate_187 = *(Tensor_542 + Intermediate_108*3 + 0);
		scalar Intermediate_188 = Intermediate_52*Intermediate_187;
		scalar Intermediate_189 = Intermediate_188+Intermediate_186;
		scalar Intermediate_190 = Intermediate_189*Intermediate_93;
		scalar Intermediate_191 = Intermediate_52*Intermediate_186;
		scalar Intermediate_192 = Intermediate_191+Intermediate_187;
		scalar Intermediate_193 = Intermediate_192*Intermediate_102;
		scalar Intermediate_194 = Intermediate_55+Intermediate_193+Intermediate_190+Intermediate_184+Intermediate_183+Intermediate_182+Intermediate_181+Intermediate_180+Intermediate_179+Intermediate_186+Intermediate_187;
		scalar Intermediate_195 = Intermediate_52*Intermediate_117*Intermediate_126;
		scalar Intermediate_196 = Intermediate_130*Intermediate_107;
		scalar Intermediate_197 = Intermediate_167*Intermediate_119;
		scalar Intermediate_198 = Intermediate_132*Intermediate_114;
		scalar Intermediate_199 = Intermediate_132*Intermediate_115;
		scalar Intermediate_200 = Intermediate_123+Intermediate_199+Intermediate_198;
		scalar Intermediate_201 = Intermediate_200*Intermediate_126;
		scalar Intermediate_202 = Intermediate_201+Intermediate_197+Intermediate_196+Intermediate_195;
		scalar Intermediate_203 = Intermediate_52*Intermediate_57*Intermediate_202*Intermediate_194*Intermediate_51;
		
		scalar Intermediate_205 = *(Tensor_544 + Intermediate_108*1 + 0);
		
		scalar Intermediate_207 = *(Tensor_547 + Intermediate_108*3 + 2);
		scalar Intermediate_208 = Intermediate_69*Intermediate_207;
		
		scalar Intermediate_210 = *(Tensor_547 + Intermediate_108*3 + 1);
		scalar Intermediate_211 = Intermediate_74*Intermediate_210;
		
		scalar Intermediate_213 = *(Tensor_547 + Intermediate_108*3 + 0);
		scalar Intermediate_214 = Intermediate_79*Intermediate_213;
		
		scalar Intermediate_216 = *(Tensor_544 + Intermediate_111*1 + 0);
		
		scalar Intermediate_218 = *(Tensor_544 + Intermediate_108*1 + 0);
		scalar Intermediate_219 = Intermediate_52*Intermediate_218;
		scalar Intermediate_220 = Intermediate_219+Intermediate_216;
		scalar Intermediate_221 = Intermediate_220*Intermediate_93;
		scalar Intermediate_222 = Intermediate_221+Intermediate_214+Intermediate_211+Intermediate_208+Intermediate_218;
		
		scalar Intermediate_224 = *(Tensor_544 + Intermediate_108*1 + 0);
		
		scalar Intermediate_226 = *(Tensor_547 + Intermediate_108*3 + 2);
		scalar Intermediate_227 = Intermediate_69*Intermediate_226;
		
		scalar Intermediate_229 = *(Tensor_547 + Intermediate_108*3 + 1);
		scalar Intermediate_230 = Intermediate_74*Intermediate_229;
		
		scalar Intermediate_232 = *(Tensor_547 + Intermediate_108*3 + 0);
		scalar Intermediate_233 = Intermediate_79*Intermediate_232;
		
		scalar Intermediate_235 = *(Tensor_544 + Intermediate_111*1 + 0);
		
		scalar Intermediate_237 = *(Tensor_544 + Intermediate_108*1 + 0);
		scalar Intermediate_238 = Intermediate_52*Intermediate_237;
		scalar Intermediate_239 = Intermediate_238+Intermediate_235;
		scalar Intermediate_240 = Intermediate_239*Intermediate_93;
		const scalar Intermediate_241 = 1.4;
		scalar Intermediate_242 = Intermediate_241+Intermediate_240+Intermediate_233+Intermediate_230+Intermediate_227+Intermediate_237;
		scalar Intermediate_243 = Intermediate_240+Intermediate_233+Intermediate_230+Intermediate_227+Intermediate_237;
		const scalar Intermediate_244 = 0.4;
		scalar Intermediate_245 = Intermediate_244*Intermediate_69*Intermediate_10;
		scalar Intermediate_246 = Intermediate_244*Intermediate_74*Intermediate_15;
		scalar Intermediate_247 = Intermediate_244*Intermediate_79*Intermediate_20;
		scalar Intermediate_248 = Intermediate_244*Intermediate_43*Intermediate_93;
		scalar Intermediate_249 = Intermediate_244*Intermediate_40;
		const scalar Intermediate_250 = 287.0;
		scalar Intermediate_251 = Intermediate_250+Intermediate_249+Intermediate_248+Intermediate_247+Intermediate_246+Intermediate_245;
		scalar Intermediate_252 = pow(Intermediate_251,Intermediate_52);
		scalar Intermediate_253 = Intermediate_252*Intermediate_243;
		scalar Intermediate_254 = Intermediate_244+Intermediate_253;
		scalar Intermediate_255 = pow(Intermediate_254,Intermediate_52);
		scalar Intermediate_256 = Intermediate_255*Intermediate_242;
		scalar Intermediate_257 = Intermediate_101+Intermediate_80+Intermediate_75+Intermediate_70+Intermediate_98;
		scalar Intermediate_258 = pow(Intermediate_257,Intermediate_132);
		scalar Intermediate_259 = Intermediate_157+Intermediate_147+Intermediate_144+Intermediate_143+Intermediate_154;
		scalar Intermediate_260 = pow(Intermediate_259,Intermediate_132);
		scalar Intermediate_261 = Intermediate_190+Intermediate_181+Intermediate_180+Intermediate_179+Intermediate_187;
		scalar Intermediate_262 = pow(Intermediate_261,Intermediate_132);
		scalar Intermediate_263 = Intermediate_55+Intermediate_262+Intermediate_260+Intermediate_258+Intermediate_256;
		scalar Intermediate_264 = Intermediate_257*Intermediate_107;
		scalar Intermediate_265 = Intermediate_259*Intermediate_119;
		scalar Intermediate_266 = Intermediate_261*Intermediate_126;
		scalar Intermediate_267 = Intermediate_266+Intermediate_265+Intermediate_264;
		scalar Intermediate_268 = Intermediate_252*Intermediate_267*Intermediate_263*Intermediate_243;
		
		scalar Intermediate_270 = *(Tensor_544 + Intermediate_111*1 + 0);
		
		scalar Intermediate_272 = *(Tensor_547 + Intermediate_111*3 + 2);
		scalar Intermediate_273 = Intermediate_83*Intermediate_272;
		
		scalar Intermediate_275 = *(Tensor_547 + Intermediate_111*3 + 1);
		scalar Intermediate_276 = Intermediate_87*Intermediate_275;
		
		scalar Intermediate_278 = *(Tensor_547 + Intermediate_111*3 + 0);
		scalar Intermediate_279 = Intermediate_91*Intermediate_278;
		scalar Intermediate_280 = *(Tensor_544 + Intermediate_111*1 + 0);
		scalar Intermediate_281 = Intermediate_52*Intermediate_280;
		scalar Intermediate_282 = Intermediate_281+Intermediate_237;
		scalar Intermediate_283 = Intermediate_282*Intermediate_102;
		scalar Intermediate_284 = Intermediate_283+Intermediate_279+Intermediate_276+Intermediate_273+Intermediate_280;
		scalar Intermediate_285 = *(Tensor_547 + Intermediate_111*3 + 2);
		scalar Intermediate_286 = Intermediate_83*Intermediate_285;
		scalar Intermediate_287 = *(Tensor_547 + Intermediate_111*3 + 1);
		scalar Intermediate_288 = Intermediate_87*Intermediate_287;
		scalar Intermediate_289 = *(Tensor_547 + Intermediate_111*3 + 0);
		scalar Intermediate_290 = Intermediate_91*Intermediate_289;
		scalar Intermediate_291 = Intermediate_52*Intermediate_280;
		scalar Intermediate_292 = Intermediate_291+Intermediate_237;
		scalar Intermediate_293 = Intermediate_292*Intermediate_102;
		scalar Intermediate_294 = Intermediate_241+Intermediate_293+Intermediate_290+Intermediate_288+Intermediate_286+Intermediate_280;
		scalar Intermediate_295 = Intermediate_293+Intermediate_290+Intermediate_288+Intermediate_286+Intermediate_280;
		scalar Intermediate_296 = Intermediate_244*Intermediate_83*Intermediate_24;
		scalar Intermediate_297 = Intermediate_244*Intermediate_87*Intermediate_28;
		scalar Intermediate_298 = Intermediate_244*Intermediate_91*Intermediate_32;
		scalar Intermediate_299 = Intermediate_244*Intermediate_48*Intermediate_102;
		scalar Intermediate_300 = Intermediate_244*Intermediate_38;
		scalar Intermediate_301 = Intermediate_250+Intermediate_300+Intermediate_299+Intermediate_298+Intermediate_297+Intermediate_296;
		scalar Intermediate_302 = pow(Intermediate_301,Intermediate_52);
		scalar Intermediate_303 = Intermediate_302*Intermediate_295;
		scalar Intermediate_304 = Intermediate_244+Intermediate_303;
		scalar Intermediate_305 = pow(Intermediate_304,Intermediate_52);
		scalar Intermediate_306 = Intermediate_305*Intermediate_294;
		scalar Intermediate_307 = Intermediate_105+Intermediate_92+Intermediate_88+Intermediate_84+Intermediate_96;
		scalar Intermediate_308 = pow(Intermediate_307,Intermediate_132);
		scalar Intermediate_309 = Intermediate_160+Intermediate_151+Intermediate_149+Intermediate_148+Intermediate_153;
		scalar Intermediate_310 = pow(Intermediate_309,Intermediate_132);
		scalar Intermediate_311 = Intermediate_193+Intermediate_184+Intermediate_183+Intermediate_182+Intermediate_186;
		scalar Intermediate_312 = pow(Intermediate_311,Intermediate_132);
		scalar Intermediate_313 = Intermediate_55+Intermediate_312+Intermediate_310+Intermediate_308+Intermediate_306;
		scalar Intermediate_314 = Intermediate_307*Intermediate_107;
		scalar Intermediate_315 = Intermediate_309*Intermediate_119;
		scalar Intermediate_316 = Intermediate_311*Intermediate_126;
		scalar Intermediate_317 = Intermediate_316+Intermediate_315+Intermediate_314;
		scalar Intermediate_318 = Intermediate_302*Intermediate_317*Intermediate_313*Intermediate_295;
		scalar Intermediate_319 = Intermediate_52*Intermediate_302*Intermediate_307*Intermediate_295;
		scalar Intermediate_320 = Intermediate_252*Intermediate_257*Intermediate_243;
		scalar Intermediate_321 = Intermediate_320+Intermediate_319;
		scalar Intermediate_322 = Intermediate_52*Intermediate_321*Intermediate_107;
		scalar Intermediate_323 = Intermediate_52*Intermediate_302*Intermediate_309*Intermediate_295;
		scalar Intermediate_324 = Intermediate_252*Intermediate_259*Intermediate_243;
		scalar Intermediate_325 = Intermediate_324+Intermediate_323;
		scalar Intermediate_326 = Intermediate_52*Intermediate_325*Intermediate_119;
		scalar Intermediate_327 = Intermediate_52*Intermediate_302*Intermediate_311*Intermediate_295;
		scalar Intermediate_328 = Intermediate_252*Intermediate_261*Intermediate_243;
		scalar Intermediate_329 = Intermediate_328+Intermediate_327;
		scalar Intermediate_330 = Intermediate_52*Intermediate_329*Intermediate_126;
		const scalar Intermediate_331 = 0.5;
		scalar Intermediate_332 = pow(Intermediate_253,Intermediate_331);
		scalar Intermediate_333 = Intermediate_332*Intermediate_257;
		scalar Intermediate_334 = pow(Intermediate_303,Intermediate_331);
		scalar Intermediate_335 = Intermediate_334*Intermediate_307;
		scalar Intermediate_336 = Intermediate_335+Intermediate_333;
		scalar Intermediate_337 = Intermediate_334+Intermediate_332;
		scalar Intermediate_338 = pow(Intermediate_337,Intermediate_52);
		scalar Intermediate_339 = Intermediate_338*Intermediate_336*Intermediate_107;
		scalar Intermediate_340 = Intermediate_332*Intermediate_259;
		scalar Intermediate_341 = Intermediate_334*Intermediate_309;
		scalar Intermediate_342 = Intermediate_341+Intermediate_340;
		scalar Intermediate_343 = Intermediate_338*Intermediate_342*Intermediate_119;
		scalar Intermediate_344 = Intermediate_332*Intermediate_261;
		scalar Intermediate_345 = Intermediate_334*Intermediate_311;
		scalar Intermediate_346 = Intermediate_345+Intermediate_344;
		scalar Intermediate_347 = Intermediate_338*Intermediate_346*Intermediate_126;
		scalar Intermediate_348 = Intermediate_347+Intermediate_343+Intermediate_339;
		scalar Intermediate_349 = Intermediate_52*Intermediate_302*Intermediate_295;
		scalar Intermediate_350 = Intermediate_253+Intermediate_349;
		scalar Intermediate_351 = Intermediate_350*Intermediate_348;
		scalar Intermediate_352 = Intermediate_351+Intermediate_330+Intermediate_326+Intermediate_322;
		
		
		scalar Intermediate_355 = pow(Intermediate_336,Intermediate_132);
		const scalar Intermediate_356 = -2;
		scalar Intermediate_357 = pow(Intermediate_337,Intermediate_356);
		scalar Intermediate_358 = Intermediate_52*Intermediate_357*Intermediate_355;
		scalar Intermediate_359 = pow(Intermediate_342,Intermediate_132);
		scalar Intermediate_360 = Intermediate_52*Intermediate_357*Intermediate_359;
		scalar Intermediate_361 = pow(Intermediate_346,Intermediate_132);
		scalar Intermediate_362 = Intermediate_52*Intermediate_357*Intermediate_361;
		scalar Intermediate_363 = Intermediate_332*Intermediate_263;
		scalar Intermediate_364 = Intermediate_334*Intermediate_313;
		scalar Intermediate_365 = Intermediate_364+Intermediate_363;
		scalar Intermediate_366 = Intermediate_338*Intermediate_365;
		const scalar Intermediate_367 = -0.1;
		scalar Intermediate_368 = Intermediate_367+Intermediate_366+Intermediate_362+Intermediate_360+Intermediate_358;
		scalar Intermediate_369 = pow(Intermediate_368,Intermediate_331);
		scalar Intermediate_370 = Intermediate_369+Intermediate_347+Intermediate_343+Intermediate_339;
		
		const scalar Intermediate_372 = 0;
		int Intermediate_373 = Intermediate_370 < Intermediate_372;
		scalar Intermediate_374 = Intermediate_52*Intermediate_338*Intermediate_336*Intermediate_107;
		scalar Intermediate_375 = Intermediate_52*Intermediate_338*Intermediate_342*Intermediate_119;
		scalar Intermediate_376 = Intermediate_52*Intermediate_338*Intermediate_346*Intermediate_126;
		scalar Intermediate_377 = Intermediate_52*Intermediate_369;
		scalar Intermediate_378 = Intermediate_377+Intermediate_376+Intermediate_375+Intermediate_374;
		
		
                scalar Intermediate_380;
                if (Intermediate_373) 
                    Intermediate_380 = Intermediate_378;
                else 
                    Intermediate_380 = Intermediate_370;
                
		
		scalar Intermediate_382 = Intermediate_52*Intermediate_257*Intermediate_107;
		scalar Intermediate_383 = Intermediate_52*Intermediate_259*Intermediate_119;
		scalar Intermediate_384 = Intermediate_52*Intermediate_261*Intermediate_126;
		scalar Intermediate_385 = Intermediate_316+Intermediate_315+Intermediate_314+Intermediate_384+Intermediate_383+Intermediate_382;
		
		int Intermediate_387 = Intermediate_385 < Intermediate_372;
		scalar Intermediate_388 = Intermediate_52*Intermediate_307*Intermediate_107;
		scalar Intermediate_389 = Intermediate_52*Intermediate_309*Intermediate_119;
		scalar Intermediate_390 = Intermediate_52*Intermediate_311*Intermediate_126;
		scalar Intermediate_391 = Intermediate_266+Intermediate_265+Intermediate_264+Intermediate_390+Intermediate_389+Intermediate_388;
		
		
                scalar Intermediate_393;
                if (Intermediate_387) 
                    Intermediate_393 = Intermediate_391;
                else 
                    Intermediate_393 = Intermediate_385;
                
		const scalar Intermediate_394 = 2.0;
		scalar Intermediate_395 = Intermediate_394*Intermediate_393;
		scalar Intermediate_396 = pow(Intermediate_243,Intermediate_52);
		scalar Intermediate_397 = Intermediate_396*Intermediate_242*Intermediate_251;
		scalar Intermediate_398 = pow(Intermediate_397,Intermediate_331);
		scalar Intermediate_399 = Intermediate_52*Intermediate_398;
		scalar Intermediate_400 = pow(Intermediate_295,Intermediate_52);
		scalar Intermediate_401 = Intermediate_400*Intermediate_294*Intermediate_301;
		scalar Intermediate_402 = pow(Intermediate_401,Intermediate_331);
		scalar Intermediate_403 = Intermediate_402+Intermediate_399;
		
		int Intermediate_405 = Intermediate_403 < Intermediate_372;
		scalar Intermediate_406 = Intermediate_52*Intermediate_402;
		scalar Intermediate_407 = Intermediate_398+Intermediate_406;
		
		
                scalar Intermediate_409;
                if (Intermediate_405) 
                    Intermediate_409 = Intermediate_407;
                else 
                    Intermediate_409 = Intermediate_403;
                
		scalar Intermediate_410 = Intermediate_394*Intermediate_409;
		scalar Intermediate_411 = Intermediate_394+Intermediate_410+Intermediate_395;
		int Intermediate_412 = Intermediate_380 < Intermediate_411;
		const scalar Intermediate_413 = 0.25;
		scalar Intermediate_414 = Intermediate_413+Intermediate_380;
		scalar Intermediate_415 = Intermediate_123+Intermediate_409+Intermediate_393;
		scalar Intermediate_416 = pow(Intermediate_415,Intermediate_52);
		scalar Intermediate_417 = Intermediate_416*Intermediate_414*Intermediate_380;
		scalar Intermediate_418 = Intermediate_123+Intermediate_417+Intermediate_409+Intermediate_393;
		
		
                scalar Intermediate_420;
                if (Intermediate_412) 
                    Intermediate_420 = Intermediate_418;
                else 
                    Intermediate_420 = Intermediate_380;
                
		scalar Intermediate_421 = Intermediate_377+Intermediate_347+Intermediate_343+Intermediate_339;
		
		int Intermediate_423 = Intermediate_421 < Intermediate_372;
		scalar Intermediate_424 = Intermediate_369+Intermediate_376+Intermediate_375+Intermediate_374;
		
		
                scalar Intermediate_426;
                if (Intermediate_423) 
                    Intermediate_426 = Intermediate_424;
                else 
                    Intermediate_426 = Intermediate_421;
                
		
		int Intermediate_428 = Intermediate_426 < Intermediate_411;
		scalar Intermediate_429 = Intermediate_413+Intermediate_426;
		scalar Intermediate_430 = Intermediate_416*Intermediate_429*Intermediate_426;
		scalar Intermediate_431 = Intermediate_123+Intermediate_430+Intermediate_409+Intermediate_393;
		
		
                scalar Intermediate_433;
                if (Intermediate_428) 
                    Intermediate_433 = Intermediate_431;
                else 
                    Intermediate_433 = Intermediate_426;
                
		scalar Intermediate_434 = Intermediate_52*Intermediate_433;
		scalar Intermediate_435 = Intermediate_55+Intermediate_434+Intermediate_420;
		scalar Intermediate_436 = -0.5;
		scalar Intermediate_437 = pow(Intermediate_368,Intermediate_436);
		scalar Intermediate_438 = Intermediate_52*Intermediate_437*Intermediate_435*Intermediate_352;
		scalar Intermediate_439 = Intermediate_52*Intermediate_302*Intermediate_313*Intermediate_295;
		scalar Intermediate_440 = Intermediate_52*Intermediate_338*Intermediate_336*Intermediate_321;
		scalar Intermediate_441 = Intermediate_52*Intermediate_338*Intermediate_342*Intermediate_325;
		scalar Intermediate_442 = Intermediate_52*Intermediate_338*Intermediate_346*Intermediate_329;
		scalar Intermediate_443 = Intermediate_252*Intermediate_263*Intermediate_243;
		scalar Intermediate_444 = Intermediate_52*Intermediate_69*Intermediate_226;
		scalar Intermediate_445 = Intermediate_52*Intermediate_74*Intermediate_229;
		scalar Intermediate_446 = Intermediate_52*Intermediate_79*Intermediate_232;
		scalar Intermediate_447 = Intermediate_52*Intermediate_239*Intermediate_93;
		scalar Intermediate_448 = Intermediate_357*Intermediate_355;
		scalar Intermediate_449 = Intermediate_357*Intermediate_359;
		scalar Intermediate_450 = Intermediate_357*Intermediate_361;
		scalar Intermediate_451 = Intermediate_55+Intermediate_450+Intermediate_449+Intermediate_448;
		scalar Intermediate_452 = Intermediate_350*Intermediate_451;
		scalar Intermediate_453 = Intermediate_244+Intermediate_238+Intermediate_293+Intermediate_452+Intermediate_290+Intermediate_288+Intermediate_286+Intermediate_447+Intermediate_446+Intermediate_445+Intermediate_444+Intermediate_443+Intermediate_442+Intermediate_441+Intermediate_440+Intermediate_439+Intermediate_280;
		
		int Intermediate_455 = Intermediate_348 < Intermediate_372;
		scalar Intermediate_456 = Intermediate_376+Intermediate_375+Intermediate_374;
		
		
                scalar Intermediate_458;
                if (Intermediate_455) 
                    Intermediate_458 = Intermediate_456;
                else 
                    Intermediate_458 = Intermediate_348;
                
		
		int Intermediate_460 = Intermediate_458 < Intermediate_411;
		scalar Intermediate_461 = Intermediate_413+Intermediate_458;
		scalar Intermediate_462 = Intermediate_416*Intermediate_461*Intermediate_458;
		scalar Intermediate_463 = Intermediate_123+Intermediate_462+Intermediate_409+Intermediate_393;
		
		
                scalar Intermediate_465;
                if (Intermediate_460) 
                    Intermediate_465 = Intermediate_463;
                else 
                    Intermediate_465 = Intermediate_458;
                
		scalar Intermediate_466 = Intermediate_52*Intermediate_465;
		scalar Intermediate_467 = Intermediate_55+Intermediate_466+Intermediate_433+Intermediate_420;
		scalar Intermediate_468 = pow(Intermediate_368,Intermediate_52);
		scalar Intermediate_469 = Intermediate_468*Intermediate_467*Intermediate_453;
		scalar Intermediate_470 = Intermediate_469+Intermediate_438;
		scalar Intermediate_471 = Intermediate_52*Intermediate_338*Intermediate_365*Intermediate_470;
		scalar Intermediate_472 = Intermediate_238+Intermediate_293+Intermediate_290+Intermediate_288+Intermediate_286+Intermediate_447+Intermediate_446+Intermediate_445+Intermediate_444+Intermediate_443+Intermediate_439+Intermediate_280;
		scalar Intermediate_473 = Intermediate_52*Intermediate_472*Intermediate_465;
		scalar Intermediate_474 = Intermediate_52*Intermediate_437*Intermediate_435*Intermediate_453;
		scalar Intermediate_475 = Intermediate_467*Intermediate_352;
		scalar Intermediate_476 = Intermediate_475+Intermediate_474;
		scalar Intermediate_477 = Intermediate_476*Intermediate_348;
		scalar Intermediate_478 = Intermediate_477+Intermediate_473+Intermediate_471+Intermediate_318+Intermediate_268+Intermediate_203+Intermediate_174+Intermediate_138+Intermediate_59;
		scalar Intermediate_479 = *(Tensor_1 + i*1 + 0);
		scalar Intermediate_480 = pow(Intermediate_479,Intermediate_52);
		scalar Intermediate_481 = Intermediate_480*Intermediate_478*Intermediate_1;
		*(Tensor_938 + Intermediate_111*1 + 0) = Intermediate_481;
		
		scalar Intermediate_483 = Intermediate_252*Intermediate_267*Intermediate_257*Intermediate_243;
		scalar Intermediate_484 = Intermediate_302*Intermediate_317*Intermediate_307*Intermediate_295;
		scalar Intermediate_485 = Intermediate_52*Intermediate_57*Intermediate_137*Intermediate_51;
		scalar Intermediate_486 = Intermediate_52*Intermediate_338*Intermediate_336*Intermediate_470;
		scalar Intermediate_487 = Intermediate_52*Intermediate_321*Intermediate_465;
		scalar Intermediate_488 = Intermediate_293+Intermediate_240+Intermediate_290+Intermediate_288+Intermediate_286+Intermediate_233+Intermediate_230+Intermediate_227+Intermediate_280+Intermediate_237;
		scalar Intermediate_489 = Intermediate_488*Intermediate_107;
		scalar Intermediate_490 = Intermediate_476*Intermediate_107;
		scalar Intermediate_491 = Intermediate_490+Intermediate_489+Intermediate_487+Intermediate_486+Intermediate_485+Intermediate_484+Intermediate_483;
		scalar Intermediate_492 = Intermediate_480*Intermediate_491*Intermediate_1;
		scalar Intermediate_493 = Intermediate_252*Intermediate_267*Intermediate_259*Intermediate_243;
		scalar Intermediate_494 = Intermediate_302*Intermediate_317*Intermediate_309*Intermediate_295;
		scalar Intermediate_495 = Intermediate_52*Intermediate_57*Intermediate_173*Intermediate_51;
		scalar Intermediate_496 = Intermediate_52*Intermediate_338*Intermediate_342*Intermediate_470;
		scalar Intermediate_497 = Intermediate_52*Intermediate_325*Intermediate_465;
		scalar Intermediate_498 = Intermediate_488*Intermediate_119;
		scalar Intermediate_499 = Intermediate_476*Intermediate_119;
		scalar Intermediate_500 = Intermediate_499+Intermediate_498+Intermediate_497+Intermediate_496+Intermediate_495+Intermediate_494+Intermediate_493;
		scalar Intermediate_501 = Intermediate_480*Intermediate_500*Intermediate_1;
		scalar Intermediate_502 = Intermediate_252*Intermediate_267*Intermediate_261*Intermediate_243;
		scalar Intermediate_503 = Intermediate_302*Intermediate_317*Intermediate_311*Intermediate_295;
		scalar Intermediate_504 = Intermediate_52*Intermediate_57*Intermediate_202*Intermediate_51;
		scalar Intermediate_505 = Intermediate_52*Intermediate_338*Intermediate_346*Intermediate_470;
		scalar Intermediate_506 = Intermediate_52*Intermediate_329*Intermediate_465;
		scalar Intermediate_507 = Intermediate_488*Intermediate_126;
		scalar Intermediate_508 = Intermediate_476*Intermediate_126;
		scalar Intermediate_509 = Intermediate_508+Intermediate_507+Intermediate_506+Intermediate_505+Intermediate_504+Intermediate_503+Intermediate_502;
		scalar Intermediate_510 = Intermediate_480*Intermediate_509*Intermediate_1;
		*(Tensor_935 + Intermediate_111*3 + 2) = Intermediate_510;
		*(Tensor_935 + Intermediate_111*3 + 2) = Intermediate_501;
		*(Tensor_935 + Intermediate_111*3 + 2) = Intermediate_492;
		
		*(Tensor_935 + Intermediate_111*3 + 2) = Intermediate_510;
		*(Tensor_935 + Intermediate_111*3 + 2) = Intermediate_501;
		*(Tensor_935 + Intermediate_111*3 + 2) = Intermediate_492;
		
		*(Tensor_935 + Intermediate_111*3 + 2) = Intermediate_510;
		*(Tensor_935 + Intermediate_111*3 + 2) = Intermediate_501;
		*(Tensor_935 + Intermediate_111*3 + 2) = Intermediate_492;
		
		scalar Intermediate_514 = Intermediate_52*Intermediate_468*Intermediate_467*Intermediate_453;
		scalar Intermediate_515 = Intermediate_252*Intermediate_267*Intermediate_243;
		scalar Intermediate_516 = Intermediate_302*Intermediate_317*Intermediate_295;
		scalar Intermediate_517 = Intermediate_437*Intermediate_435*Intermediate_352;
		scalar Intermediate_518 = Intermediate_52*Intermediate_350*Intermediate_465;
		scalar Intermediate_519 = Intermediate_518+Intermediate_517+Intermediate_516+Intermediate_515+Intermediate_514;
		scalar Intermediate_520 = Intermediate_480*Intermediate_519*Intermediate_1;
		*(Tensor_932 + Intermediate_111*1 + 0) = Intermediate_520;
		
	}
	long long end = current_timestamp(); mil += end-start; printf("c module Function_characteristicFlux: %lld\n", mil);
}

void Function_coupledFlux(int n, const scalar* Tensor_939, const scalar* Tensor_940, const scalar* Tensor_941, const scalar* Tensor_942, const scalar* Tensor_943, const scalar* Tensor_944, const scalar* Tensor_0, const scalar* Tensor_1, const scalar* Tensor_2, const scalar* Tensor_3, const scalar* Tensor_4, const scalar* Tensor_5, const scalar* Tensor_6, const scalar* Tensor_7, const integer* Tensor_8, const integer* Tensor_9, scalar* Tensor_1329, scalar* Tensor_1332, scalar* Tensor_1335) {
	long long start = current_timestamp();
	for (integer i = 0; i < n; i++) {
		integer Intermediate_0 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_1 = *(Tensor_0 + i*1 + 0);
		integer Intermediate_2 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_4 = *(Tensor_940 + Intermediate_2*1 + 0);
		integer Intermediate_5 = *(Tensor_8 + i*1 + 0);
		
		scalar Intermediate_7 = *(Tensor_940 + Intermediate_5*1 + 0);
		integer Intermediate_8 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_10 = *(Tensor_943 + Intermediate_8*3 + 2);
		scalar Intermediate_11 = *(Tensor_7 + i*6 + 5);
		scalar Intermediate_12 = Intermediate_11*Intermediate_10;
		integer Intermediate_13 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_15 = *(Tensor_943 + Intermediate_13*3 + 1);
		scalar Intermediate_16 = *(Tensor_7 + i*6 + 4);
		scalar Intermediate_17 = Intermediate_16*Intermediate_15;
		integer Intermediate_18 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_20 = *(Tensor_943 + Intermediate_18*3 + 0);
		scalar Intermediate_21 = *(Tensor_7 + i*6 + 3);
		scalar Intermediate_22 = Intermediate_21*Intermediate_20;
		integer Intermediate_23 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_24 = *(Tensor_943 + Intermediate_23*3 + 2);
		scalar Intermediate_25 = *(Tensor_7 + i*6 + 2);
		scalar Intermediate_26 = Intermediate_25*Intermediate_24;
		integer Intermediate_27 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_28 = *(Tensor_943 + Intermediate_27*3 + 1);
		scalar Intermediate_29 = *(Tensor_7 + i*6 + 1);
		scalar Intermediate_30 = Intermediate_29*Intermediate_28;
		integer Intermediate_31 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_32 = *(Tensor_943 + Intermediate_31*3 + 0);
		scalar Intermediate_33 = *(Tensor_7 + i*6 + 0);
		scalar Intermediate_34 = Intermediate_33*Intermediate_32;
		scalar Intermediate_35 = *(Tensor_6 + i*2 + 1);
		integer Intermediate_36 = *(Tensor_8 + i*1 + 0);
		
		scalar Intermediate_38 = *(Tensor_940 + Intermediate_36*1 + 0);
		integer Intermediate_39 = *(Tensor_9 + i*1 + 0);
		scalar Intermediate_40 = *(Tensor_940 + Intermediate_39*1 + 0);
		const scalar Intermediate_41 = -1;
		scalar Intermediate_42 = Intermediate_41*Intermediate_40;
		scalar Intermediate_43 = Intermediate_42+Intermediate_38;
		scalar Intermediate_44 = Intermediate_43*Intermediate_35;
		scalar Intermediate_45 = *(Tensor_6 + i*2 + 0);
		const scalar Intermediate_46 = -1;
		scalar Intermediate_47 = Intermediate_46*Intermediate_38;
		scalar Intermediate_48 = Intermediate_47+Intermediate_40;
		scalar Intermediate_49 = Intermediate_48*Intermediate_45;
		const scalar Intermediate_50 = 0.500025;
		scalar Intermediate_51 = Intermediate_50+Intermediate_49+Intermediate_44+Intermediate_34+Intermediate_30+Intermediate_26+Intermediate_22+Intermediate_17+Intermediate_12+Intermediate_38+Intermediate_40;
		const scalar Intermediate_52 = -1;
		scalar Intermediate_53 = *(Tensor_4 + i*1 + 0);
		scalar Intermediate_54 = pow(Intermediate_53,Intermediate_52);
		const scalar Intermediate_55 = 0.5;
		scalar Intermediate_56 = Intermediate_55+Intermediate_49+Intermediate_44+Intermediate_34+Intermediate_30+Intermediate_26+Intermediate_22+Intermediate_17+Intermediate_12+Intermediate_38+Intermediate_40;
		scalar Intermediate_57 = pow(Intermediate_56,Intermediate_52);
		const scalar Intermediate_58 = -1435.0;
		scalar Intermediate_59 = Intermediate_58*Intermediate_57*Intermediate_54*Intermediate_48*Intermediate_51;
		integer Intermediate_60 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_62 = *(Tensor_939 + Intermediate_60*3 + 1);
		integer Intermediate_63 = *(Tensor_8 + i*1 + 0);
		
		scalar Intermediate_65 = *(Tensor_939 + Intermediate_63*3 + 1);
		integer Intermediate_66 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_68 = *(Tensor_942 + Intermediate_66*9 + 5);
		scalar Intermediate_69 = *(Tensor_7 + i*6 + 5);
		scalar Intermediate_70 = Intermediate_69*Intermediate_68;
		integer Intermediate_71 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_73 = *(Tensor_942 + Intermediate_71*9 + 4);
		scalar Intermediate_74 = *(Tensor_7 + i*6 + 4);
		scalar Intermediate_75 = Intermediate_74*Intermediate_73;
		integer Intermediate_76 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_78 = *(Tensor_942 + Intermediate_76*9 + 3);
		scalar Intermediate_79 = *(Tensor_7 + i*6 + 3);
		scalar Intermediate_80 = Intermediate_79*Intermediate_78;
		integer Intermediate_81 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_82 = *(Tensor_942 + Intermediate_81*9 + 5);
		scalar Intermediate_83 = *(Tensor_7 + i*6 + 2);
		scalar Intermediate_84 = Intermediate_83*Intermediate_82;
		integer Intermediate_85 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_86 = *(Tensor_942 + Intermediate_85*9 + 4);
		scalar Intermediate_87 = *(Tensor_7 + i*6 + 1);
		scalar Intermediate_88 = Intermediate_87*Intermediate_86;
		integer Intermediate_89 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_90 = *(Tensor_942 + Intermediate_89*9 + 3);
		scalar Intermediate_91 = *(Tensor_7 + i*6 + 0);
		scalar Intermediate_92 = Intermediate_91*Intermediate_90;
		scalar Intermediate_93 = *(Tensor_6 + i*2 + 1);
		integer Intermediate_94 = *(Tensor_8 + i*1 + 0);
		
		scalar Intermediate_96 = *(Tensor_939 + Intermediate_94*3 + 1);
		integer Intermediate_97 = *(Tensor_9 + i*1 + 0);
		scalar Intermediate_98 = *(Tensor_939 + Intermediate_97*3 + 1);
		scalar Intermediate_99 = Intermediate_52*Intermediate_98;
		scalar Intermediate_100 = Intermediate_99+Intermediate_96;
		scalar Intermediate_101 = Intermediate_100*Intermediate_93;
		scalar Intermediate_102 = *(Tensor_6 + i*2 + 0);
		scalar Intermediate_103 = Intermediate_52*Intermediate_96;
		scalar Intermediate_104 = Intermediate_103+Intermediate_98;
		scalar Intermediate_105 = Intermediate_104*Intermediate_102;
		scalar Intermediate_106 = Intermediate_55+Intermediate_105+Intermediate_101+Intermediate_92+Intermediate_88+Intermediate_84+Intermediate_80+Intermediate_75+Intermediate_70+Intermediate_96+Intermediate_98;
		scalar Intermediate_107 = *(Tensor_5 + i*3 + 1);
		integer Intermediate_108 = *(Tensor_9 + i*1 + 0);
		
		scalar Intermediate_110 = *(Tensor_942 + Intermediate_108*9 + 0);
		integer Intermediate_111 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_112 = *(Tensor_942 + Intermediate_111*9 + 0);
		
		scalar Intermediate_114 = *(Tensor_942 + Intermediate_108*9 + 8);
		scalar Intermediate_115 = *(Tensor_942 + Intermediate_111*9 + 8);
		const scalar Intermediate_116 = 2.16666666666667;
		scalar Intermediate_117 = Intermediate_116+Intermediate_115+Intermediate_114+Intermediate_112+Intermediate_110+Intermediate_86+Intermediate_73;
		scalar Intermediate_118 = Intermediate_52*Intermediate_117*Intermediate_107;
		scalar Intermediate_119 = *(Tensor_5 + i*3 + 0);
		
		scalar Intermediate_121 = *(Tensor_942 + Intermediate_108*9 + 1);
		scalar Intermediate_122 = *(Tensor_942 + Intermediate_111*9 + 1);
		const scalar Intermediate_123 = 1.0;
		scalar Intermediate_124 = Intermediate_123+Intermediate_122+Intermediate_121+Intermediate_90+Intermediate_78;
		scalar Intermediate_125 = Intermediate_124*Intermediate_119;
		scalar Intermediate_126 = *(Tensor_5 + i*3 + 2);
		
		scalar Intermediate_128 = *(Tensor_942 + Intermediate_108*9 + 7);
		scalar Intermediate_129 = *(Tensor_942 + Intermediate_111*9 + 7);
		scalar Intermediate_130 = Intermediate_123+Intermediate_129+Intermediate_128+Intermediate_82+Intermediate_68;
		scalar Intermediate_131 = Intermediate_130*Intermediate_126;
		const scalar Intermediate_132 = 2;
		scalar Intermediate_133 = Intermediate_132*Intermediate_73;
		scalar Intermediate_134 = Intermediate_132*Intermediate_86;
		scalar Intermediate_135 = Intermediate_123+Intermediate_134+Intermediate_133;
		scalar Intermediate_136 = Intermediate_135*Intermediate_107;
		scalar Intermediate_137 = Intermediate_136+Intermediate_131+Intermediate_125+Intermediate_118;
		scalar Intermediate_138 = Intermediate_52*Intermediate_57*Intermediate_137*Intermediate_106*Intermediate_51;
		
		scalar Intermediate_140 = *(Tensor_939 + Intermediate_108*3 + 0);
		
		scalar Intermediate_142 = *(Tensor_939 + Intermediate_111*3 + 0);
		
		scalar Intermediate_144 = *(Tensor_942 + Intermediate_108*9 + 2);
		scalar Intermediate_145 = Intermediate_69*Intermediate_144;
		scalar Intermediate_146 = Intermediate_74*Intermediate_121;
		scalar Intermediate_147 = Intermediate_79*Intermediate_110;
		scalar Intermediate_148 = *(Tensor_942 + Intermediate_111*9 + 2);
		scalar Intermediate_149 = Intermediate_83*Intermediate_148;
		scalar Intermediate_150 = Intermediate_87*Intermediate_122;
		scalar Intermediate_151 = Intermediate_91*Intermediate_112;
		
		scalar Intermediate_153 = *(Tensor_939 + Intermediate_111*3 + 0);
		scalar Intermediate_154 = *(Tensor_939 + Intermediate_108*3 + 0);
		scalar Intermediate_155 = Intermediate_52*Intermediate_154;
		scalar Intermediate_156 = Intermediate_155+Intermediate_153;
		scalar Intermediate_157 = Intermediate_156*Intermediate_93;
		scalar Intermediate_158 = Intermediate_52*Intermediate_153;
		scalar Intermediate_159 = Intermediate_158+Intermediate_154;
		scalar Intermediate_160 = Intermediate_159*Intermediate_102;
		scalar Intermediate_161 = Intermediate_55+Intermediate_160+Intermediate_157+Intermediate_151+Intermediate_150+Intermediate_149+Intermediate_147+Intermediate_146+Intermediate_145+Intermediate_153+Intermediate_154;
		scalar Intermediate_162 = Intermediate_52*Intermediate_117*Intermediate_119;
		scalar Intermediate_163 = Intermediate_124*Intermediate_107;
		
		scalar Intermediate_165 = *(Tensor_942 + Intermediate_108*9 + 6);
		scalar Intermediate_166 = *(Tensor_942 + Intermediate_111*9 + 6);
		scalar Intermediate_167 = Intermediate_123+Intermediate_166+Intermediate_165+Intermediate_148+Intermediate_144;
		scalar Intermediate_168 = Intermediate_167*Intermediate_126;
		scalar Intermediate_169 = Intermediate_132*Intermediate_110;
		scalar Intermediate_170 = Intermediate_132*Intermediate_112;
		scalar Intermediate_171 = Intermediate_123+Intermediate_170+Intermediate_169;
		scalar Intermediate_172 = Intermediate_171*Intermediate_119;
		scalar Intermediate_173 = Intermediate_172+Intermediate_168+Intermediate_163+Intermediate_162;
		scalar Intermediate_174 = Intermediate_52*Intermediate_57*Intermediate_173*Intermediate_161*Intermediate_51;
		
		scalar Intermediate_176 = *(Tensor_939 + Intermediate_108*3 + 2);
		
		scalar Intermediate_178 = *(Tensor_939 + Intermediate_111*3 + 2);
		scalar Intermediate_179 = Intermediate_69*Intermediate_114;
		scalar Intermediate_180 = Intermediate_74*Intermediate_128;
		scalar Intermediate_181 = Intermediate_79*Intermediate_165;
		scalar Intermediate_182 = Intermediate_83*Intermediate_115;
		scalar Intermediate_183 = Intermediate_87*Intermediate_129;
		scalar Intermediate_184 = Intermediate_91*Intermediate_166;
		
		scalar Intermediate_186 = *(Tensor_939 + Intermediate_111*3 + 2);
		scalar Intermediate_187 = *(Tensor_939 + Intermediate_108*3 + 2);
		scalar Intermediate_188 = Intermediate_52*Intermediate_187;
		scalar Intermediate_189 = Intermediate_188+Intermediate_186;
		scalar Intermediate_190 = Intermediate_189*Intermediate_93;
		scalar Intermediate_191 = Intermediate_52*Intermediate_186;
		scalar Intermediate_192 = Intermediate_191+Intermediate_187;
		scalar Intermediate_193 = Intermediate_192*Intermediate_102;
		scalar Intermediate_194 = Intermediate_55+Intermediate_193+Intermediate_190+Intermediate_184+Intermediate_183+Intermediate_182+Intermediate_181+Intermediate_180+Intermediate_179+Intermediate_186+Intermediate_187;
		scalar Intermediate_195 = Intermediate_52*Intermediate_117*Intermediate_126;
		scalar Intermediate_196 = Intermediate_130*Intermediate_107;
		scalar Intermediate_197 = Intermediate_167*Intermediate_119;
		scalar Intermediate_198 = Intermediate_132*Intermediate_114;
		scalar Intermediate_199 = Intermediate_132*Intermediate_115;
		scalar Intermediate_200 = Intermediate_123+Intermediate_199+Intermediate_198;
		scalar Intermediate_201 = Intermediate_200*Intermediate_126;
		scalar Intermediate_202 = Intermediate_201+Intermediate_197+Intermediate_196+Intermediate_195;
		scalar Intermediate_203 = Intermediate_52*Intermediate_57*Intermediate_202*Intermediate_194*Intermediate_51;
		
		scalar Intermediate_205 = *(Tensor_941 + Intermediate_108*1 + 0);
		
		scalar Intermediate_207 = *(Tensor_944 + Intermediate_108*3 + 2);
		scalar Intermediate_208 = Intermediate_69*Intermediate_207;
		
		scalar Intermediate_210 = *(Tensor_944 + Intermediate_108*3 + 1);
		scalar Intermediate_211 = Intermediate_74*Intermediate_210;
		
		scalar Intermediate_213 = *(Tensor_944 + Intermediate_108*3 + 0);
		scalar Intermediate_214 = Intermediate_79*Intermediate_213;
		
		scalar Intermediate_216 = *(Tensor_941 + Intermediate_111*1 + 0);
		
		scalar Intermediate_218 = *(Tensor_941 + Intermediate_108*1 + 0);
		scalar Intermediate_219 = Intermediate_52*Intermediate_218;
		scalar Intermediate_220 = Intermediate_219+Intermediate_216;
		scalar Intermediate_221 = Intermediate_220*Intermediate_93;
		scalar Intermediate_222 = Intermediate_221+Intermediate_214+Intermediate_211+Intermediate_208+Intermediate_218;
		
		scalar Intermediate_224 = *(Tensor_941 + Intermediate_108*1 + 0);
		
		scalar Intermediate_226 = *(Tensor_944 + Intermediate_108*3 + 2);
		scalar Intermediate_227 = Intermediate_69*Intermediate_226;
		
		scalar Intermediate_229 = *(Tensor_944 + Intermediate_108*3 + 1);
		scalar Intermediate_230 = Intermediate_74*Intermediate_229;
		
		scalar Intermediate_232 = *(Tensor_944 + Intermediate_108*3 + 0);
		scalar Intermediate_233 = Intermediate_79*Intermediate_232;
		
		scalar Intermediate_235 = *(Tensor_941 + Intermediate_111*1 + 0);
		
		scalar Intermediate_237 = *(Tensor_941 + Intermediate_108*1 + 0);
		scalar Intermediate_238 = Intermediate_52*Intermediate_237;
		scalar Intermediate_239 = Intermediate_238+Intermediate_235;
		scalar Intermediate_240 = Intermediate_239*Intermediate_93;
		const scalar Intermediate_241 = 1.4;
		scalar Intermediate_242 = Intermediate_241+Intermediate_240+Intermediate_233+Intermediate_230+Intermediate_227+Intermediate_237;
		scalar Intermediate_243 = Intermediate_240+Intermediate_233+Intermediate_230+Intermediate_227+Intermediate_237;
		const scalar Intermediate_244 = 0.4;
		scalar Intermediate_245 = Intermediate_244*Intermediate_69*Intermediate_10;
		scalar Intermediate_246 = Intermediate_244*Intermediate_74*Intermediate_15;
		scalar Intermediate_247 = Intermediate_244*Intermediate_79*Intermediate_20;
		scalar Intermediate_248 = Intermediate_244*Intermediate_43*Intermediate_93;
		scalar Intermediate_249 = Intermediate_244*Intermediate_40;
		const scalar Intermediate_250 = 287.0;
		scalar Intermediate_251 = Intermediate_250+Intermediate_249+Intermediate_248+Intermediate_247+Intermediate_246+Intermediate_245;
		scalar Intermediate_252 = pow(Intermediate_251,Intermediate_52);
		scalar Intermediate_253 = Intermediate_252*Intermediate_243;
		scalar Intermediate_254 = Intermediate_244+Intermediate_253;
		scalar Intermediate_255 = pow(Intermediate_254,Intermediate_52);
		scalar Intermediate_256 = Intermediate_255*Intermediate_242;
		scalar Intermediate_257 = Intermediate_190+Intermediate_181+Intermediate_180+Intermediate_179+Intermediate_187;
		scalar Intermediate_258 = pow(Intermediate_257,Intermediate_132);
		scalar Intermediate_259 = Intermediate_101+Intermediate_80+Intermediate_75+Intermediate_70+Intermediate_98;
		scalar Intermediate_260 = pow(Intermediate_259,Intermediate_132);
		scalar Intermediate_261 = Intermediate_157+Intermediate_147+Intermediate_146+Intermediate_145+Intermediate_154;
		scalar Intermediate_262 = pow(Intermediate_261,Intermediate_132);
		scalar Intermediate_263 = Intermediate_55+Intermediate_262+Intermediate_260+Intermediate_258+Intermediate_256;
		scalar Intermediate_264 = Intermediate_257*Intermediate_126;
		scalar Intermediate_265 = Intermediate_259*Intermediate_107;
		scalar Intermediate_266 = Intermediate_261*Intermediate_119;
		scalar Intermediate_267 = Intermediate_266+Intermediate_265+Intermediate_264;
		scalar Intermediate_268 = Intermediate_252*Intermediate_267*Intermediate_263*Intermediate_243;
		
		scalar Intermediate_270 = *(Tensor_941 + Intermediate_111*1 + 0);
		
		scalar Intermediate_272 = *(Tensor_944 + Intermediate_111*3 + 2);
		scalar Intermediate_273 = Intermediate_83*Intermediate_272;
		
		scalar Intermediate_275 = *(Tensor_944 + Intermediate_111*3 + 1);
		scalar Intermediate_276 = Intermediate_87*Intermediate_275;
		
		scalar Intermediate_278 = *(Tensor_944 + Intermediate_111*3 + 0);
		scalar Intermediate_279 = Intermediate_91*Intermediate_278;
		scalar Intermediate_280 = *(Tensor_941 + Intermediate_111*1 + 0);
		scalar Intermediate_281 = Intermediate_52*Intermediate_280;
		scalar Intermediate_282 = Intermediate_281+Intermediate_237;
		scalar Intermediate_283 = Intermediate_282*Intermediate_102;
		scalar Intermediate_284 = Intermediate_283+Intermediate_279+Intermediate_276+Intermediate_273+Intermediate_280;
		scalar Intermediate_285 = *(Tensor_944 + Intermediate_111*3 + 2);
		scalar Intermediate_286 = Intermediate_83*Intermediate_285;
		scalar Intermediate_287 = *(Tensor_944 + Intermediate_111*3 + 1);
		scalar Intermediate_288 = Intermediate_87*Intermediate_287;
		scalar Intermediate_289 = *(Tensor_944 + Intermediate_111*3 + 0);
		scalar Intermediate_290 = Intermediate_91*Intermediate_289;
		scalar Intermediate_291 = Intermediate_52*Intermediate_280;
		scalar Intermediate_292 = Intermediate_291+Intermediate_237;
		scalar Intermediate_293 = Intermediate_292*Intermediate_102;
		scalar Intermediate_294 = Intermediate_241+Intermediate_293+Intermediate_290+Intermediate_288+Intermediate_286+Intermediate_280;
		scalar Intermediate_295 = Intermediate_293+Intermediate_290+Intermediate_288+Intermediate_286+Intermediate_280;
		scalar Intermediate_296 = Intermediate_244*Intermediate_83*Intermediate_24;
		scalar Intermediate_297 = Intermediate_244*Intermediate_87*Intermediate_28;
		scalar Intermediate_298 = Intermediate_244*Intermediate_91*Intermediate_32;
		scalar Intermediate_299 = Intermediate_244*Intermediate_48*Intermediate_102;
		scalar Intermediate_300 = Intermediate_244*Intermediate_38;
		scalar Intermediate_301 = Intermediate_250+Intermediate_300+Intermediate_299+Intermediate_298+Intermediate_297+Intermediate_296;
		scalar Intermediate_302 = pow(Intermediate_301,Intermediate_52);
		scalar Intermediate_303 = Intermediate_302*Intermediate_295;
		scalar Intermediate_304 = Intermediate_244+Intermediate_303;
		scalar Intermediate_305 = pow(Intermediate_304,Intermediate_52);
		scalar Intermediate_306 = Intermediate_305*Intermediate_294;
		scalar Intermediate_307 = Intermediate_193+Intermediate_184+Intermediate_183+Intermediate_182+Intermediate_186;
		scalar Intermediate_308 = pow(Intermediate_307,Intermediate_132);
		scalar Intermediate_309 = Intermediate_105+Intermediate_92+Intermediate_88+Intermediate_84+Intermediate_96;
		scalar Intermediate_310 = pow(Intermediate_309,Intermediate_132);
		scalar Intermediate_311 = Intermediate_160+Intermediate_151+Intermediate_150+Intermediate_149+Intermediate_153;
		scalar Intermediate_312 = pow(Intermediate_311,Intermediate_132);
		scalar Intermediate_313 = Intermediate_55+Intermediate_312+Intermediate_310+Intermediate_308+Intermediate_306;
		scalar Intermediate_314 = Intermediate_307*Intermediate_126;
		scalar Intermediate_315 = Intermediate_309*Intermediate_107;
		scalar Intermediate_316 = Intermediate_311*Intermediate_119;
		scalar Intermediate_317 = Intermediate_316+Intermediate_315+Intermediate_314;
		scalar Intermediate_318 = Intermediate_302*Intermediate_317*Intermediate_313*Intermediate_295;
		scalar Intermediate_319 = Intermediate_52*Intermediate_302*Intermediate_307*Intermediate_295;
		scalar Intermediate_320 = Intermediate_252*Intermediate_257*Intermediate_243;
		scalar Intermediate_321 = Intermediate_320+Intermediate_319;
		scalar Intermediate_322 = Intermediate_52*Intermediate_321*Intermediate_126;
		scalar Intermediate_323 = Intermediate_52*Intermediate_302*Intermediate_309*Intermediate_295;
		scalar Intermediate_324 = Intermediate_252*Intermediate_259*Intermediate_243;
		scalar Intermediate_325 = Intermediate_324+Intermediate_323;
		scalar Intermediate_326 = Intermediate_52*Intermediate_325*Intermediate_107;
		scalar Intermediate_327 = Intermediate_52*Intermediate_302*Intermediate_311*Intermediate_295;
		scalar Intermediate_328 = Intermediate_252*Intermediate_261*Intermediate_243;
		scalar Intermediate_329 = Intermediate_328+Intermediate_327;
		scalar Intermediate_330 = Intermediate_52*Intermediate_329*Intermediate_119;
		const scalar Intermediate_331 = 0.5;
		scalar Intermediate_332 = pow(Intermediate_253,Intermediate_331);
		scalar Intermediate_333 = Intermediate_332*Intermediate_257;
		scalar Intermediate_334 = pow(Intermediate_303,Intermediate_331);
		scalar Intermediate_335 = Intermediate_334*Intermediate_307;
		scalar Intermediate_336 = Intermediate_335+Intermediate_333;
		scalar Intermediate_337 = Intermediate_334+Intermediate_332;
		scalar Intermediate_338 = pow(Intermediate_337,Intermediate_52);
		scalar Intermediate_339 = Intermediate_338*Intermediate_336*Intermediate_126;
		scalar Intermediate_340 = Intermediate_332*Intermediate_259;
		scalar Intermediate_341 = Intermediate_334*Intermediate_309;
		scalar Intermediate_342 = Intermediate_341+Intermediate_340;
		scalar Intermediate_343 = Intermediate_338*Intermediate_342*Intermediate_107;
		scalar Intermediate_344 = Intermediate_332*Intermediate_261;
		scalar Intermediate_345 = Intermediate_334*Intermediate_311;
		scalar Intermediate_346 = Intermediate_345+Intermediate_344;
		scalar Intermediate_347 = Intermediate_338*Intermediate_346*Intermediate_119;
		scalar Intermediate_348 = Intermediate_347+Intermediate_343+Intermediate_339;
		scalar Intermediate_349 = Intermediate_52*Intermediate_302*Intermediate_295;
		scalar Intermediate_350 = Intermediate_253+Intermediate_349;
		scalar Intermediate_351 = Intermediate_350*Intermediate_348;
		scalar Intermediate_352 = Intermediate_351+Intermediate_330+Intermediate_326+Intermediate_322;
		
		
		scalar Intermediate_355 = pow(Intermediate_336,Intermediate_132);
		const scalar Intermediate_356 = -2;
		scalar Intermediate_357 = pow(Intermediate_337,Intermediate_356);
		scalar Intermediate_358 = Intermediate_52*Intermediate_357*Intermediate_355;
		scalar Intermediate_359 = pow(Intermediate_342,Intermediate_132);
		scalar Intermediate_360 = Intermediate_52*Intermediate_357*Intermediate_359;
		scalar Intermediate_361 = pow(Intermediate_346,Intermediate_132);
		scalar Intermediate_362 = Intermediate_52*Intermediate_357*Intermediate_361;
		scalar Intermediate_363 = Intermediate_332*Intermediate_263;
		scalar Intermediate_364 = Intermediate_334*Intermediate_313;
		scalar Intermediate_365 = Intermediate_364+Intermediate_363;
		scalar Intermediate_366 = Intermediate_338*Intermediate_365;
		const scalar Intermediate_367 = -0.1;
		scalar Intermediate_368 = Intermediate_367+Intermediate_366+Intermediate_362+Intermediate_360+Intermediate_358;
		scalar Intermediate_369 = pow(Intermediate_368,Intermediate_331);
		scalar Intermediate_370 = Intermediate_369+Intermediate_347+Intermediate_343+Intermediate_339;
		
		const scalar Intermediate_372 = 0;
		int Intermediate_373 = Intermediate_370 < Intermediate_372;
		scalar Intermediate_374 = Intermediate_52*Intermediate_338*Intermediate_336*Intermediate_126;
		scalar Intermediate_375 = Intermediate_52*Intermediate_338*Intermediate_342*Intermediate_107;
		scalar Intermediate_376 = Intermediate_52*Intermediate_338*Intermediate_346*Intermediate_119;
		scalar Intermediate_377 = Intermediate_52*Intermediate_369;
		scalar Intermediate_378 = Intermediate_377+Intermediate_376+Intermediate_375+Intermediate_374;
		
		
                scalar Intermediate_380;
                if (Intermediate_373) 
                    Intermediate_380 = Intermediate_378;
                else 
                    Intermediate_380 = Intermediate_370;
                
		
		scalar Intermediate_382 = Intermediate_52*Intermediate_257*Intermediate_126;
		scalar Intermediate_383 = Intermediate_52*Intermediate_259*Intermediate_107;
		scalar Intermediate_384 = Intermediate_52*Intermediate_261*Intermediate_119;
		scalar Intermediate_385 = Intermediate_316+Intermediate_315+Intermediate_314+Intermediate_384+Intermediate_383+Intermediate_382;
		
		int Intermediate_387 = Intermediate_385 < Intermediate_372;
		scalar Intermediate_388 = Intermediate_52*Intermediate_307*Intermediate_126;
		scalar Intermediate_389 = Intermediate_52*Intermediate_309*Intermediate_107;
		scalar Intermediate_390 = Intermediate_52*Intermediate_311*Intermediate_119;
		scalar Intermediate_391 = Intermediate_266+Intermediate_265+Intermediate_264+Intermediate_390+Intermediate_389+Intermediate_388;
		
		
                scalar Intermediate_393;
                if (Intermediate_387) 
                    Intermediate_393 = Intermediate_391;
                else 
                    Intermediate_393 = Intermediate_385;
                
		const scalar Intermediate_394 = 2.0;
		scalar Intermediate_395 = Intermediate_394*Intermediate_393;
		scalar Intermediate_396 = pow(Intermediate_243,Intermediate_52);
		scalar Intermediate_397 = Intermediate_396*Intermediate_242*Intermediate_251;
		scalar Intermediate_398 = pow(Intermediate_397,Intermediate_331);
		scalar Intermediate_399 = Intermediate_52*Intermediate_398;
		scalar Intermediate_400 = pow(Intermediate_295,Intermediate_52);
		scalar Intermediate_401 = Intermediate_400*Intermediate_294*Intermediate_301;
		scalar Intermediate_402 = pow(Intermediate_401,Intermediate_331);
		scalar Intermediate_403 = Intermediate_402+Intermediate_399;
		
		int Intermediate_405 = Intermediate_403 < Intermediate_372;
		scalar Intermediate_406 = Intermediate_52*Intermediate_402;
		scalar Intermediate_407 = Intermediate_398+Intermediate_406;
		
		
                scalar Intermediate_409;
                if (Intermediate_405) 
                    Intermediate_409 = Intermediate_407;
                else 
                    Intermediate_409 = Intermediate_403;
                
		scalar Intermediate_410 = Intermediate_394*Intermediate_409;
		scalar Intermediate_411 = Intermediate_394+Intermediate_410+Intermediate_395;
		int Intermediate_412 = Intermediate_380 < Intermediate_411;
		const scalar Intermediate_413 = 0.25;
		scalar Intermediate_414 = Intermediate_413+Intermediate_380;
		scalar Intermediate_415 = Intermediate_123+Intermediate_409+Intermediate_393;
		scalar Intermediate_416 = pow(Intermediate_415,Intermediate_52);
		scalar Intermediate_417 = Intermediate_416*Intermediate_414*Intermediate_380;
		scalar Intermediate_418 = Intermediate_123+Intermediate_417+Intermediate_409+Intermediate_393;
		
		
                scalar Intermediate_420;
                if (Intermediate_412) 
                    Intermediate_420 = Intermediate_418;
                else 
                    Intermediate_420 = Intermediate_380;
                
		scalar Intermediate_421 = Intermediate_377+Intermediate_347+Intermediate_343+Intermediate_339;
		
		int Intermediate_423 = Intermediate_421 < Intermediate_372;
		scalar Intermediate_424 = Intermediate_369+Intermediate_376+Intermediate_375+Intermediate_374;
		
		
                scalar Intermediate_426;
                if (Intermediate_423) 
                    Intermediate_426 = Intermediate_424;
                else 
                    Intermediate_426 = Intermediate_421;
                
		
		int Intermediate_428 = Intermediate_426 < Intermediate_411;
		scalar Intermediate_429 = Intermediate_413+Intermediate_426;
		scalar Intermediate_430 = Intermediate_416*Intermediate_429*Intermediate_426;
		scalar Intermediate_431 = Intermediate_123+Intermediate_430+Intermediate_409+Intermediate_393;
		
		
                scalar Intermediate_433;
                if (Intermediate_428) 
                    Intermediate_433 = Intermediate_431;
                else 
                    Intermediate_433 = Intermediate_426;
                
		scalar Intermediate_434 = Intermediate_52*Intermediate_433;
		scalar Intermediate_435 = Intermediate_55+Intermediate_434+Intermediate_420;
		scalar Intermediate_436 = -0.5;
		scalar Intermediate_437 = pow(Intermediate_368,Intermediate_436);
		scalar Intermediate_438 = Intermediate_52*Intermediate_437*Intermediate_435*Intermediate_352;
		scalar Intermediate_439 = Intermediate_52*Intermediate_302*Intermediate_313*Intermediate_295;
		scalar Intermediate_440 = Intermediate_52*Intermediate_338*Intermediate_336*Intermediate_321;
		scalar Intermediate_441 = Intermediate_52*Intermediate_338*Intermediate_342*Intermediate_325;
		scalar Intermediate_442 = Intermediate_52*Intermediate_338*Intermediate_346*Intermediate_329;
		scalar Intermediate_443 = Intermediate_252*Intermediate_263*Intermediate_243;
		scalar Intermediate_444 = Intermediate_52*Intermediate_69*Intermediate_226;
		scalar Intermediate_445 = Intermediate_52*Intermediate_74*Intermediate_229;
		scalar Intermediate_446 = Intermediate_52*Intermediate_79*Intermediate_232;
		scalar Intermediate_447 = Intermediate_52*Intermediate_239*Intermediate_93;
		scalar Intermediate_448 = Intermediate_357*Intermediate_355;
		scalar Intermediate_449 = Intermediate_357*Intermediate_359;
		scalar Intermediate_450 = Intermediate_357*Intermediate_361;
		scalar Intermediate_451 = Intermediate_55+Intermediate_450+Intermediate_449+Intermediate_448;
		scalar Intermediate_452 = Intermediate_350*Intermediate_451;
		scalar Intermediate_453 = Intermediate_244+Intermediate_238+Intermediate_293+Intermediate_452+Intermediate_290+Intermediate_288+Intermediate_286+Intermediate_447+Intermediate_446+Intermediate_445+Intermediate_444+Intermediate_443+Intermediate_442+Intermediate_441+Intermediate_440+Intermediate_439+Intermediate_280;
		
		int Intermediate_455 = Intermediate_348 < Intermediate_372;
		scalar Intermediate_456 = Intermediate_376+Intermediate_375+Intermediate_374;
		
		
                scalar Intermediate_458;
                if (Intermediate_455) 
                    Intermediate_458 = Intermediate_456;
                else 
                    Intermediate_458 = Intermediate_348;
                
		
		int Intermediate_460 = Intermediate_458 < Intermediate_411;
		scalar Intermediate_461 = Intermediate_413+Intermediate_458;
		scalar Intermediate_462 = Intermediate_416*Intermediate_461*Intermediate_458;
		scalar Intermediate_463 = Intermediate_123+Intermediate_462+Intermediate_409+Intermediate_393;
		
		
                scalar Intermediate_465;
                if (Intermediate_460) 
                    Intermediate_465 = Intermediate_463;
                else 
                    Intermediate_465 = Intermediate_458;
                
		scalar Intermediate_466 = Intermediate_52*Intermediate_465;
		scalar Intermediate_467 = Intermediate_55+Intermediate_466+Intermediate_433+Intermediate_420;
		scalar Intermediate_468 = pow(Intermediate_368,Intermediate_52);
		scalar Intermediate_469 = Intermediate_468*Intermediate_467*Intermediate_453;
		scalar Intermediate_470 = Intermediate_469+Intermediate_438;
		scalar Intermediate_471 = Intermediate_52*Intermediate_338*Intermediate_365*Intermediate_470;
		scalar Intermediate_472 = Intermediate_238+Intermediate_293+Intermediate_290+Intermediate_288+Intermediate_286+Intermediate_447+Intermediate_446+Intermediate_445+Intermediate_444+Intermediate_443+Intermediate_439+Intermediate_280;
		scalar Intermediate_473 = Intermediate_52*Intermediate_472*Intermediate_465;
		scalar Intermediate_474 = Intermediate_52*Intermediate_437*Intermediate_435*Intermediate_453;
		scalar Intermediate_475 = Intermediate_467*Intermediate_352;
		scalar Intermediate_476 = Intermediate_475+Intermediate_474;
		scalar Intermediate_477 = Intermediate_476*Intermediate_348;
		scalar Intermediate_478 = Intermediate_477+Intermediate_473+Intermediate_471+Intermediate_318+Intermediate_268+Intermediate_203+Intermediate_174+Intermediate_138+Intermediate_59;
		scalar Intermediate_479 = *(Tensor_1 + i*1 + 0);
		scalar Intermediate_480 = pow(Intermediate_479,Intermediate_52);
		scalar Intermediate_481 = Intermediate_480*Intermediate_478*Intermediate_1;
		*(Tensor_1335 + Intermediate_111*1 + 0) = Intermediate_481;
		
		scalar Intermediate_483 = Intermediate_252*Intermediate_267*Intermediate_257*Intermediate_243;
		scalar Intermediate_484 = Intermediate_302*Intermediate_317*Intermediate_307*Intermediate_295;
		scalar Intermediate_485 = Intermediate_52*Intermediate_57*Intermediate_202*Intermediate_51;
		scalar Intermediate_486 = Intermediate_52*Intermediate_338*Intermediate_336*Intermediate_470;
		scalar Intermediate_487 = Intermediate_52*Intermediate_321*Intermediate_465;
		scalar Intermediate_488 = Intermediate_293+Intermediate_240+Intermediate_290+Intermediate_288+Intermediate_286+Intermediate_233+Intermediate_230+Intermediate_227+Intermediate_280+Intermediate_237;
		scalar Intermediate_489 = Intermediate_488*Intermediate_126;
		scalar Intermediate_490 = Intermediate_476*Intermediate_126;
		scalar Intermediate_491 = Intermediate_490+Intermediate_489+Intermediate_487+Intermediate_486+Intermediate_485+Intermediate_484+Intermediate_483;
		scalar Intermediate_492 = Intermediate_480*Intermediate_491*Intermediate_1;
		scalar Intermediate_493 = Intermediate_252*Intermediate_267*Intermediate_259*Intermediate_243;
		scalar Intermediate_494 = Intermediate_302*Intermediate_317*Intermediate_309*Intermediate_295;
		scalar Intermediate_495 = Intermediate_52*Intermediate_57*Intermediate_137*Intermediate_51;
		scalar Intermediate_496 = Intermediate_52*Intermediate_338*Intermediate_342*Intermediate_470;
		scalar Intermediate_497 = Intermediate_52*Intermediate_325*Intermediate_465;
		scalar Intermediate_498 = Intermediate_488*Intermediate_107;
		scalar Intermediate_499 = Intermediate_476*Intermediate_107;
		scalar Intermediate_500 = Intermediate_499+Intermediate_498+Intermediate_497+Intermediate_496+Intermediate_495+Intermediate_494+Intermediate_493;
		scalar Intermediate_501 = Intermediate_480*Intermediate_500*Intermediate_1;
		scalar Intermediate_502 = Intermediate_252*Intermediate_267*Intermediate_261*Intermediate_243;
		scalar Intermediate_503 = Intermediate_302*Intermediate_317*Intermediate_311*Intermediate_295;
		scalar Intermediate_504 = Intermediate_52*Intermediate_57*Intermediate_173*Intermediate_51;
		scalar Intermediate_505 = Intermediate_52*Intermediate_338*Intermediate_346*Intermediate_470;
		scalar Intermediate_506 = Intermediate_52*Intermediate_329*Intermediate_465;
		scalar Intermediate_507 = Intermediate_488*Intermediate_119;
		scalar Intermediate_508 = Intermediate_476*Intermediate_119;
		scalar Intermediate_509 = Intermediate_508+Intermediate_507+Intermediate_506+Intermediate_505+Intermediate_504+Intermediate_503+Intermediate_502;
		scalar Intermediate_510 = Intermediate_480*Intermediate_509*Intermediate_1;
		*(Tensor_1332 + Intermediate_111*3 + 2) = Intermediate_510;
		*(Tensor_1332 + Intermediate_111*3 + 2) = Intermediate_501;
		*(Tensor_1332 + Intermediate_111*3 + 2) = Intermediate_492;
		
		*(Tensor_1332 + Intermediate_111*3 + 2) = Intermediate_510;
		*(Tensor_1332 + Intermediate_111*3 + 2) = Intermediate_501;
		*(Tensor_1332 + Intermediate_111*3 + 2) = Intermediate_492;
		
		*(Tensor_1332 + Intermediate_111*3 + 2) = Intermediate_510;
		*(Tensor_1332 + Intermediate_111*3 + 2) = Intermediate_501;
		*(Tensor_1332 + Intermediate_111*3 + 2) = Intermediate_492;
		
		scalar Intermediate_514 = Intermediate_52*Intermediate_468*Intermediate_467*Intermediate_453;
		scalar Intermediate_515 = Intermediate_252*Intermediate_267*Intermediate_243;
		scalar Intermediate_516 = Intermediate_302*Intermediate_317*Intermediate_295;
		scalar Intermediate_517 = Intermediate_437*Intermediate_435*Intermediate_352;
		scalar Intermediate_518 = Intermediate_52*Intermediate_350*Intermediate_465;
		scalar Intermediate_519 = Intermediate_518+Intermediate_517+Intermediate_516+Intermediate_515+Intermediate_514;
		scalar Intermediate_520 = Intermediate_480*Intermediate_519*Intermediate_1;
		*(Tensor_1329 + Intermediate_111*1 + 0) = Intermediate_520;
		
	}
	long long end = current_timestamp(); mil += end-start; printf("c module Function_coupledFlux: %lld\n", mil);
}

void Function_boundaryFlux(int n, const scalar* Tensor_1336, const scalar* Tensor_1337, const scalar* Tensor_1338, const scalar* Tensor_1339, const scalar* Tensor_1340, const scalar* Tensor_1341, const scalar* Tensor_0, const scalar* Tensor_1, const scalar* Tensor_2, const scalar* Tensor_3, const scalar* Tensor_4, const scalar* Tensor_5, const scalar* Tensor_6, const scalar* Tensor_7, const integer* Tensor_8, const integer* Tensor_9, scalar* Tensor_1459, scalar* Tensor_1462, scalar* Tensor_1465) {
	long long start = current_timestamp();
	for (integer i = 0; i < n; i++) {
		integer Intermediate_0 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_1 = *(Tensor_0 + i*1 + 0);
		integer Intermediate_2 = *(Tensor_9 + i*1 + 0);
		scalar Intermediate_3 = *(Tensor_1337 + i*1 + 0);
		scalar Intermediate_4 = *(Tensor_1337 + Intermediate_2*1 + 0);
		integer Intermediate_5 = *(Tensor_8 + i*1 + 0);
		scalar Intermediate_6 = *(Tensor_1337 + Intermediate_5*1 + 0);
		const scalar Intermediate_7 = -1;
		scalar Intermediate_8 = Intermediate_7*Intermediate_6;
		scalar Intermediate_9 = Intermediate_8+Intermediate_4;
		const scalar Intermediate_10 = 2.5e-5;
		scalar Intermediate_11 = Intermediate_10+Intermediate_4;
		scalar Intermediate_12 = pow(Intermediate_4,Intermediate_7);
		scalar Intermediate_13 = *(Tensor_4 + i*1 + 0);
		scalar Intermediate_14 = pow(Intermediate_13,Intermediate_7);
		const scalar Intermediate_15 = -1435.0;
		scalar Intermediate_16 = Intermediate_15*Intermediate_14*Intermediate_12*Intermediate_11*Intermediate_9;
		integer Intermediate_17 = *(Tensor_9 + i*1 + 0);
		scalar Intermediate_18 = *(Tensor_1336 + i*3 + 2);
		scalar Intermediate_19 = *(Tensor_1336 + Intermediate_17*3 + 2);
		scalar Intermediate_20 = *(Tensor_1339 + i*9 + 8);
		scalar Intermediate_21 = *(Tensor_1339 + Intermediate_17*9 + 8);
		scalar Intermediate_22 = *(Tensor_5 + i*3 + 2);
		const scalar Intermediate_23 = 2;
		scalar Intermediate_24 = Intermediate_23*Intermediate_22*Intermediate_21;
		scalar Intermediate_25 = *(Tensor_1339 + i*9 + 4);
		scalar Intermediate_26 = *(Tensor_1339 + Intermediate_17*9 + 4);
		scalar Intermediate_27 = *(Tensor_1339 + i*9 + 0);
		scalar Intermediate_28 = *(Tensor_1339 + Intermediate_17*9 + 0);
		const scalar Intermediate_29 = 0.666666666666667;
		scalar Intermediate_30 = Intermediate_29+Intermediate_28+Intermediate_26+Intermediate_21;
		scalar Intermediate_31 = Intermediate_7*Intermediate_30*Intermediate_22;
		scalar Intermediate_32 = *(Tensor_5 + i*3 + 1);
		scalar Intermediate_33 = *(Tensor_1339 + i*9 + 7);
		scalar Intermediate_34 = *(Tensor_1339 + Intermediate_17*9 + 7);
		scalar Intermediate_35 = *(Tensor_1339 + i*9 + 5);
		scalar Intermediate_36 = *(Tensor_1339 + Intermediate_17*9 + 5);
		scalar Intermediate_37 = Intermediate_36+Intermediate_34;
		scalar Intermediate_38 = Intermediate_37*Intermediate_32;
		scalar Intermediate_39 = *(Tensor_5 + i*3 + 0);
		scalar Intermediate_40 = *(Tensor_1339 + i*9 + 6);
		scalar Intermediate_41 = *(Tensor_1339 + Intermediate_17*9 + 6);
		scalar Intermediate_42 = *(Tensor_1339 + i*9 + 2);
		scalar Intermediate_43 = *(Tensor_1339 + Intermediate_17*9 + 2);
		scalar Intermediate_44 = Intermediate_43+Intermediate_41;
		scalar Intermediate_45 = Intermediate_44*Intermediate_39;
		scalar Intermediate_46 = Intermediate_45+Intermediate_38+Intermediate_31+Intermediate_24;
		scalar Intermediate_47 = Intermediate_7*Intermediate_12*Intermediate_11*Intermediate_46*Intermediate_19;
		scalar Intermediate_48 = *(Tensor_1336 + i*3 + 0);
		scalar Intermediate_49 = *(Tensor_1336 + Intermediate_17*3 + 0);
		scalar Intermediate_50 = Intermediate_23*Intermediate_39*Intermediate_28;
		scalar Intermediate_51 = Intermediate_7*Intermediate_30*Intermediate_39;
		scalar Intermediate_52 = Intermediate_44*Intermediate_22;
		scalar Intermediate_53 = *(Tensor_1339 + i*9 + 3);
		scalar Intermediate_54 = *(Tensor_1339 + Intermediate_17*9 + 3);
		scalar Intermediate_55 = *(Tensor_1339 + i*9 + 1);
		scalar Intermediate_56 = *(Tensor_1339 + Intermediate_17*9 + 1);
		scalar Intermediate_57 = Intermediate_56+Intermediate_54;
		scalar Intermediate_58 = Intermediate_57*Intermediate_32;
		scalar Intermediate_59 = Intermediate_58+Intermediate_52+Intermediate_51+Intermediate_50;
		scalar Intermediate_60 = Intermediate_7*Intermediate_12*Intermediate_11*Intermediate_59*Intermediate_49;
		scalar Intermediate_61 = *(Tensor_1336 + i*3 + 1);
		scalar Intermediate_62 = *(Tensor_1336 + Intermediate_17*3 + 1);
		scalar Intermediate_63 = Intermediate_23*Intermediate_32*Intermediate_26;
		scalar Intermediate_64 = Intermediate_7*Intermediate_30*Intermediate_32;
		scalar Intermediate_65 = Intermediate_37*Intermediate_22;
		scalar Intermediate_66 = Intermediate_57*Intermediate_39;
		scalar Intermediate_67 = Intermediate_66+Intermediate_65+Intermediate_64+Intermediate_63;
		scalar Intermediate_68 = Intermediate_7*Intermediate_12*Intermediate_11*Intermediate_67*Intermediate_62;
		scalar Intermediate_69 = Intermediate_22*Intermediate_19;
		scalar Intermediate_70 = Intermediate_32*Intermediate_62;
		scalar Intermediate_71 = Intermediate_39*Intermediate_49;
		scalar Intermediate_72 = Intermediate_71+Intermediate_70+Intermediate_69;
		scalar Intermediate_73 = *(Tensor_1338 + i*1 + 0);
		scalar Intermediate_74 = *(Tensor_1338 + Intermediate_17*1 + 0);
		scalar Intermediate_75 = *(Tensor_1338 + i*1 + 0);
		scalar Intermediate_76 = *(Tensor_1338 + Intermediate_17*1 + 0);
		scalar Intermediate_77 = pow(Intermediate_19,Intermediate_23);
		scalar Intermediate_78 = pow(Intermediate_62,Intermediate_23);
		scalar Intermediate_79 = pow(Intermediate_49,Intermediate_23);
		const scalar Intermediate_80 = 718.0;
		scalar Intermediate_81 = Intermediate_80+Intermediate_79+Intermediate_78+Intermediate_77+Intermediate_4;
		const scalar Intermediate_82 = 0.4;
		scalar Intermediate_83 = Intermediate_82*Intermediate_4;
		const scalar Intermediate_84 = 287.0;
		scalar Intermediate_85 = Intermediate_84+Intermediate_83;
		scalar Intermediate_86 = pow(Intermediate_85,Intermediate_7);
		scalar Intermediate_87 = Intermediate_86*Intermediate_81*Intermediate_76;
		scalar Intermediate_88 = Intermediate_87+Intermediate_76;
		scalar Intermediate_89 = Intermediate_88*Intermediate_72;
		scalar Intermediate_90 = Intermediate_89+Intermediate_68+Intermediate_60+Intermediate_47+Intermediate_16;
		scalar Intermediate_91 = *(Tensor_1 + i*1 + 0);
		scalar Intermediate_92 = pow(Intermediate_91,Intermediate_7);
		scalar Intermediate_93 = Intermediate_92*Intermediate_90*Intermediate_1;
		*(Tensor_1465 + Intermediate_5*1 + 0) = Intermediate_93;
		
		scalar Intermediate_95 = Intermediate_86*Intermediate_72*Intermediate_19*Intermediate_76;
		scalar Intermediate_96 = Intermediate_7*Intermediate_12*Intermediate_11*Intermediate_46;
		scalar Intermediate_97 = Intermediate_22*Intermediate_76;
		scalar Intermediate_98 = Intermediate_97+Intermediate_96+Intermediate_95;
		scalar Intermediate_99 = Intermediate_92*Intermediate_98*Intermediate_1;
		scalar Intermediate_100 = Intermediate_86*Intermediate_72*Intermediate_62*Intermediate_76;
		scalar Intermediate_101 = Intermediate_7*Intermediate_12*Intermediate_11*Intermediate_67;
		scalar Intermediate_102 = Intermediate_32*Intermediate_76;
		scalar Intermediate_103 = Intermediate_102+Intermediate_101+Intermediate_100;
		scalar Intermediate_104 = Intermediate_92*Intermediate_103*Intermediate_1;
		scalar Intermediate_105 = Intermediate_86*Intermediate_72*Intermediate_49*Intermediate_76;
		scalar Intermediate_106 = Intermediate_7*Intermediate_12*Intermediate_11*Intermediate_59;
		scalar Intermediate_107 = Intermediate_39*Intermediate_76;
		scalar Intermediate_108 = Intermediate_107+Intermediate_106+Intermediate_105;
		scalar Intermediate_109 = Intermediate_92*Intermediate_108*Intermediate_1;
		*(Tensor_1462 + Intermediate_5*3 + 2) = Intermediate_109;
		*(Tensor_1462 + Intermediate_5*3 + 2) = Intermediate_104;
		*(Tensor_1462 + Intermediate_5*3 + 2) = Intermediate_99;
		
		*(Tensor_1462 + Intermediate_5*3 + 2) = Intermediate_109;
		*(Tensor_1462 + Intermediate_5*3 + 2) = Intermediate_104;
		*(Tensor_1462 + Intermediate_5*3 + 2) = Intermediate_99;
		
		*(Tensor_1462 + Intermediate_5*3 + 2) = Intermediate_109;
		*(Tensor_1462 + Intermediate_5*3 + 2) = Intermediate_104;
		*(Tensor_1462 + Intermediate_5*3 + 2) = Intermediate_99;
		
		scalar Intermediate_113 = Intermediate_86*Intermediate_92*Intermediate_72*Intermediate_1*Intermediate_76;
		*(Tensor_1459 + Intermediate_5*1 + 0) = Intermediate_113;
		
	}
	long long end = current_timestamp(); mil += end-start; printf("c module Function_boundaryFlux: %lld\n", mil);
}
