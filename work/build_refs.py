# -*- coding: utf-8 -*-
"""Final reference builder v2: manual strings for all 46, keeping verified DOIs."""
import json

# key -> (vancouver string, identifier for verification)
REFS = [
("cox1972", "Cox DR. Regression models and life-tables. J R Stat Soc Ser B Stat Methodol. 1972;34(2):187-220. doi:10.1111/j.2517-6161.1972.tb00899.x"),
("harrell1996", "Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. Stat Med. 1996;15(4):361-387. doi:10.1002/(SICI)1097-0258(19960229)15:4<361::AID-SIM168>3.0.CO;2-4"),
("tcga2013", "Weinstein JN, Collisson EA, Mills GB, Shaw KR, Ozenberger BA, Ellrott K, et al. The Cancer Genome Atlas Pan-Cancer analysis project. Nat Genet. 2013;45(10):1113-1120. doi:10.1038/ng.2764"),
("xena2020", "Goldman MJ, Craft B, Hastie M, Repečka K, McDade F, Kamath A, et al. Visualizing and interpreting cancer genomics data via the Xena platform. Nat Biotechnol. 2020;38(6):675-678. doi:10.1038/s41587-020-0546-8"),
("geo2013", "Barrett T, Wilhite SE, Ledoux P, Evangelista C, Kim IF, Tomashevsky M, et al. NCBI GEO: archive for functional genomics data sets, update. Nucleic Acids Res. 2013;41(D1):D991-D995. doi:10.1093/nar/gks1193"),
("geoquery2007", "Davis S, Meltzer PS. GEOquery: a bridge between the Gene Expression Omnibus (GEO) and BioConductor. Bioinformatics. 2007;23(14):1846-1847. doi:10.1093/bioinformatics/btm254"),
("glmnet2010", "Friedman J, Hastie T, Tibshirani R. Regularization paths for generalized linear models via coordinate descent. J Stat Softw. 2010;33(1):1-22. doi:10.18637/jss.v033.i01"),
("glmnetcox2011", "Simon N, Friedman J, Hastie T, Tibshirani R. Regularization paths for Cox's proportional hazards model via coordinate descent. J Stat Softw. 2011;39(5):1-13. doi:10.18637/jss.v039.i05"),
("deepsurv2018", "Katzman JL, Shaham U, Cloninger A, Bates J, Jiang T, Kluger Y. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. BMC Med Res Methodol. 2018;18(1):24. doi:10.1186/s12874-018-0482-1"),
("coxnnet2018", "Ching T, Zhu X, Garmire LX. Cox-nnet: an artificial neural network method for prognosis prediction of high-throughput omics data. PLoS Comput Biol. 2018;14(4):e1006076. doi:10.1371/journal.pcbi.1006076"),
("deephit2018", "Lee C, Zame W, Yoon J, van der Schaar M. DeepHit: a deep learning approach to survival analysis with competing risks. Proc AAAI Conf Artif Intell. 2018;32(1):2314-2321. doi:10.1609/aaai.v32i1.11842"),
("rsf2008", "Ishwaran H, Kogalur UB, Blackstone EH, Lauer MS. Random survival forests. Ann Appl Stat. 2008;2(3):841-860. doi:10.1214/08-AOAS169"),
("eraslan2019", "Eraslan G, Avsec Ž, Gagneur J, Theis FJ. Deep learning: new computational modelling techniques for genomics. Nat Rev Genet. 2019;20(7):389-403. doi:10.1038/s41576-019-0122-6"),
("wainberg2018", "Wainberg M, Merico D, Delong A, Frey BJ. Deep learning in biomedicine. Nat Biotechnol. 2018;36(9):829-838. doi:10.1038/nbt.4233"),
("miotto2018", "Miotto R, Wang F, Wang S, Jiang X, Dudley JT. Deep learning for healthcare: review, opportunities and challenges. Brief Bioinform. 2018;19(6):1236-1246. doi:10.1093/bib/bbx044"),
("gcn2017", "Kipf TN, Welling M. Semi-supervised classification with graph convolutional networks. arXiv:1609.02907 [cs.LG]. 2017."),
("gat2018", "Veličković P, Cucurull G, Casanova A, Romero A, Liò P, Bengio Y. Graph attention networks. arXiv:1710.10903 [cs.LG]. 2018."),
("zitnik2018", "Zitnik M, Agrawal M, Leskovec J. Modeling polypharmacy side effects with graph convolutional networks. Bioinformatics. 2018;34(13):i457-i466. doi:10.1093/bioinformatics/bty294"),
("kegg2023", "Kanehisa M, Furumichi M, Sato Y, Kawashima M, Ishiguro-Watanabe M. KEGG for taxonomy-based analysis of pathways and genomes. Nucleic Acids Res. 2023;51(D1):D587-D592. doi:10.1093/nar/gkac963"),
("pathgnn2022", "Liang J, Zhang Y, Chen H, et al. Risk stratification and pathway analysis based on graph neural network and interpretable algorithm. BMC Bioinformatics. 2022;23(1):411. doi:10.1186/s12859-022-04950-1"),
("coxpath2024", "Ma T, Zhao H, Zhao Q, Wang J. Cox-Path: biological pathway-informed graph neural network for cancer survival prediction. In: Proceedings of the 15th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics (ACM-BCB 2024). New York: ACM; 2024. doi:10.1145/3698587.3701397"),
("priorgnn2024", "Yan H, Weng D, Li D, Gu Y, Ma W, Liu Q. Prior knowledge-guided multilevel graph neural network for tumor risk prediction and interpretation via multi-omics data integration. Brief Bioinform. 2024;25(4):bbae184. doi:10.1093/bib/bbae184"),
("pathmog2026", "Wang Z, et al. PathMoG: a pathway-centric modular graph neural network for multi-omics survival prediction. arXiv:2604.24371 [q-bio.QM]. 2026."),
("dingo2015", "Ha MJ, Baladandayuthapani V. DINGO: differential network analysis in genomics. Bioinformatics. 2015;31(21):3413-3420. doi:10.1093/bioinformatics/btv406"),
("delfuente2010", "de la Fuente A. From differential expression to differential networking: identification of dysfunctional regulatory networks in diseases. Trends Genet. 2010;26(7):326-333. doi:10.1016/j.tig.2010.05.001"),
("gill2010", "Gill R, Datta S, Datta S. A statistical framework for differential network analysis from microarray data. BMC Bioinformatics. 2010;11:95. doi:10.1186/1471-2105-11-95"),
("tcga_biolinks2016", "Colaprico A, Silva TC, Olsen C, Garofano L, Cava C, Garolini D, et al. TCGAbiolinks: an R/Bioconductor package for integrative analysis of TCGA data. Nucleic Acids Res. 2016;44(8):e71. doi:10.1093/nar/gkv1507"),
("therneau2000", "Therneau TM, Grambsch PM. Modeling Survival Data: Extending the Cox Model. New York: Springer; 2000. doi:10.1007/978-1-4757-3294-8"),
("bh1995", "Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. J R Stat Soc Ser B Stat Methodol. 1995;57(1):289-300. doi:10.1111/j.2517-6161.1995.tb02031.x"),
("uno2011", "Uno H, Cai T, Pencina MJ, D'Agostino RB, Wei LJ. On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data. Stat Med. 2011;30(10):1105-1117. doi:10.1002/sim.4154"),
("heagerty2005", "Heagerty PJ, Zheng Y. Survival model predictive accuracy and ROC curves. Biometrics. 2005;61(1):92-105. doi:10.1111/j.0006-341X.2005.030814.x"),
("vickers2006", "Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006;26(6):565-574. doi:10.1177/0272989X06295361"),
("altman2009", "Altman DG, Vergouwe Y, Royston P, Moons KG. Prognosis and prognostic research: validating a prognostic model. BMJ. 2009;338:b605. doi:10.1136/bmj.b605"),
("crowson2016", "Crowson CS, Atkinson EJ, Therneau TM. Assessing calibration of prognostic risk scores. Stat Methods Med Res. 2016;25(4):1692-1706. doi:10.1177/0962280213497434"),
("pytorch2019", "Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, et al. PyTorch: an imperative style, high-performance deep learning library. arXiv:1912.01703 [cs.LG]. 2019."),
("sklearn2011", "Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: machine learning in Python. arXiv:1201.0490 [cs.LG]. 2011."),
("lifelines2019", "Davidson-Pilon C. lifelines: survival analysis in Python. J Open Source Softw. 2019;4(39):1317. doi:10.21105/joss.01317"),
("gsva2013", "Hänzelmann S, Castelo R, Guinney J. GSVA: gene set variation analysis for microarray and RNA-seq data. BMC Bioinformatics. 2013;14:7. doi:10.1186/1471-2105-14-7"),
("hallmark2015", "Liberzon A, Birger C, Thorvaldsdóttir H, Ghandi M, Mesirov JP, Tamayo P. The Molecular Signatures Database Hallmark gene set collection. Cell Syst. 2015;1(6):417-425. doi:10.1016/j.cels.2015.12.004"),
("mcpcounter2016", "Becht E, Giraldo NA, Lacroix L, Buttard B, Elarouci N, Petitprez F, et al. Estimating the population abundance of tissue-infiltrating immune and stromal cell populations using gene expression. Genome Biol. 2016;17:218. doi:10.1186/s13059-016-1070-5"),
("estimate2013", "Yoshihara K, Shahmoradgoli M, Martínez E, Vegesna R, Kim H, Torres-Garcia W, et al. Inferring tumour purity and stromal and immune cell admixture from expression data. Nat Commun. 2013;4:2612. doi:10.1038/ncomms3612"),
("tmb2017", "Chalmers ZR, Connelly CF, Fabrizio D, Gay L, Ali SM, Ennis R, et al. Analysis of 100,000 human cancer genomes reveals the landscape of tumor mutational burden. Genome Med. 2017;9:34. doi:10.1186/s13073-017-0424-2"),
("imvigor2018", "Mariathasan S, Turley SJ, Nickles D, et al. TGFβ attenuates tumour response to PD-L1 blockade by contributing to exclusion of T cells. Nature. 2018;554(7693):544-548. doi:10.1038/nature25501"),
("gdsc2013", "Yang W, Soares J, Greninger P, Edelman EJ, Lightfoot H, Forbes S, et al. Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells. Nucleic Acids Res. 2013;41(D1):D955-D961. doi:10.1093/nar/gks1111"),
("oncopredict2021", "Maeser D, Gruener RF, Huang RS. oncoPredict: an R package for predicting in vivo or cancer patient drug response and biomarkers from cell line screening data. Brief Bioinform. 2021;22(6):bbab260. doi:10.1093/bib/bbab260"),
("metabric2012", "Curtis C, Shah SP, Chin SF, Turashvili G, Rueda OM, Dunning MJ, et al. The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups. Nature. 2012;486(7403):346-352. doi:10.1038/nature10983"),
]
assert len(REFS) == 46, len(REFS)
json.dump({k: {"n": i+1, "text": s} for i, (k, s) in enumerate(REFS)},
          open("work/refs_final.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for i, (k, s) in enumerate(REFS, 1):
    print(i, s[:130])
