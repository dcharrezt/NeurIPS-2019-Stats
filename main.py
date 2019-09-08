s = """
      <p><b>Multimodal Model-Agnostic Meta-Learning via Task-Aware Modulation</b><br><i>Risto Vuorio (University of Michigan) &middot; Shao-Hua Sun (University of Southern California) &middot; Hexiang Hu (University of Southern California) &middot; Joseph J Lim (University of Southern California)</i></p>

      <p><b>ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks</b><br><i>Jiasen Lu (Georgia Tech) &middot; Dhruv Batra (Georgia Tech / Facebook AI Research (FAIR)) &middot; Devi Parikh (Georgia Tech / Facebook AI Research (FAIR)) &middot; Stefan Lee (Georgia Institute of Technology)</i></p>

      <p><b>Stochastic Shared Embeddings: Data-driven Regularization of Embedding Layers</b><br><i>Liwei Wu (University of California, Davis) &middot; Shuqing Li (University of California, Davis) &middot; Cho-Jui Hsieh (UCLA) &middot; James Sharpnack (UC Davis)</i></p>

      <p><b>Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video</b><br><i>JiaWang Bian (The University of Adelaide) &middot; Zhichao Li (Tusimple) &middot; Naiyan Wang (Hong Kong University of Science and Technology) &middot; Huangying Zhan (The University of Adelaide) &middot; Chunhua Shen (University of Adelaide) &middot; Ming-Ming Cheng (Nankai University) &middot; Ian Reid (University of Adelaide)</i></p>

      <p><b>Zero-shot Learning via Simultaneous Generating and Learning</b><br><i>Hyeonwoo Yu (Seoul National University) &middot; Beomhee Lee (Seoul National University)</i></p>

      <p><b>Ask not what AI can do for you, but what AI should do: Towards a framework of task delegability</b><br><i>Brian Lubars (University of Colorado Boulder) &middot; Chenhao Tan (University of Colorado Boulder)</i></p>

      <p><b>Stand-Alone Self-Attention in Vision Models</b><br><i>Niki Parmar (Google) &middot; Prajit Ramachandran (Google Brain) &middot; Ashish Vaswani (Google Brain) &middot; Irwan Bello (Google) &middot; Anselm Levskaya (Google) &middot; Jon Shlens (Google Research)</i></p>

      <p><b>High Fidelity Video Prediction with Large Neural Nets</b><br><i>Ruben Villegas (Adobe Research / U. Michigan) &middot; Arkanath Pathak (Google) &middot; Harini Kannan (Google Brain) &middot; Honglak Lee (Google / U. Michigan) &middot; Dumitru Erhan (Google Brain) &middot; Quoc V Le (Google)</i></p>

      <p><b>Unsupervised learning of object structure and dynamics from videos</b><br><i>Matthias Minderer (Google Research) &middot; Chen Sun (Google Research) &middot; Ruben Villegas (Adobe Research / U. Michigan) &middot; Forrester Cole (Google Research) &middot; Kevin P Murphy (Google) &middot; Honglak Lee (Google Brain)</i></p>

      <p><b>TensorPipe: Easy Scaling with Micro-Batch Pipeline Parallelism</b><br><i>Yanping Huang (Google Brain) &middot; Youlong Cheng (Google) &middot; Ankur Bapna (Google) &middot; Orhan Firat (Google) &middot; Dehao Chen (Google) &middot; Mia Chen (Google Brain) &middot; HyoukJoong Lee (Google) &middot; Jiquan Ngiam (Google Brain) &middot; Quoc V Le (Google) &middot; Yonghui Wu (Google) &middot; zhifeng Chen (Google Brain)</i></p>

      <p><b>Meta-Learning with Implicit Gradients</b><br><i>Aravind Rajeswaran (University of Washington) &middot; Chelsea Finn (Stanford University) &middot; Sham Kakade (University of Washington) &middot; Sergey Levine (UC Berkeley)</i></p>

      <p><b>Adversarial Examples Are Not Bugs, They Are Features</b><br><i>Andrew Ilyas (MIT) &middot; Shibani Santurkar (MIT) &middot; Dimitris Tsipras (MIT) &middot; Logan Engstrom (MIT) &middot; Brandon Tran (Massachusetts Institute of Technology) &middot; Aleksander Madry (MIT)</i></p>

      <p><b>Social-BiGAT: Multimodal Trajectory Forecasting using Bicycle-GAN and Graph Attention Networks</b><br><i>Vineet Kosaraju (Stanford University) &middot; Amir Sadeghian (Stanford University) &middot; Roberto Martín-Martín (Stanford University) &middot; Ian Reid (University of Adelaide) &middot; Hamid Rezatofighi (University of Adelaide) &middot; Silvio Savarese (Stanford University)</i></p>

      <p><b>FreeAnchor: Learning to Match Anchors for Visual Object Detection</b><br><i>Xiaosong Zhang (University of Chinese Academy of Sciences) &middot; Fang Wan (University of Chinese Academy of Sciences) &middot; Chang Liu (University of Chinese Academy of Sciences) &middot; Rongrong Ji (Xiamen University, China) &middot; Qixiang Ye (University of Chinese Academy of Sciences, China)</i></p>

      <p><b>Differentially Private Hypothesis Selection</b><br><i>Mark Bun (Princeton University) &middot; Gautam Kamath (University of Waterloo) &middot; Thomas Steinke (IBM, Almaden) &middot; Steven Wu (Microsoft Research)</i></p>

      <p><b>New Differentially Private Algorithms for Learning Mixtures of Well-Separated Gaussians</b><br><i>Gautam Kamath (University of Waterloo) &middot; Or Sheffet (University of Alberta) &middot; Vikrant Singhal (Northeastern University) &middot; Jonathan Ullman (Northeastern University)</i></p>

      <p><b>Average-Case Averages: Private Algorithms for Smooth Sensitivity and Mean Estimation</b><br><i>Mark Bun (Princeton University) &middot; Thomas Steinke (IBM, Almaden)</i></p>

      <p><b>Multi-Resolution Weak Supervision for Sequential Data</b><br><i>Paroma Varma (Stanford University) &middot; Frederic Sala (Stanford) &middot; Shiori Sagawa (Stanford University) &middot; Jason Fries (Stanford University) &middot; Daniel Fu (Stanford University) &middot; Saelig Khattar (Stanford University) &middot; Ashwini Ramamoorthy (Stanford University) &middot; Ke Xiao (Stanford University) &middot; Kayvon  Fatahalian (Stanford) &middot; James Priest (Stanford University) &middot; Christopher Ré (Stanford)</i></p>

      <p><b>DeepUSPS: Deep Robust Unsupervised Saliency Prediction via Self-supervision</b><br><i>Tam Nguyen (Freiburg Computer Vision Lab) &middot; Maximilian Dax (Bosch GmbH) &middot; Chaithanya Kumar Mummadi (Robert Bosch GmbH) &middot; Nhung Ngo (Bosch Center for Artificial Intelligence) &middot; Thi Hoai Phuong Nguyen (KIT) &middot; Zhongyu Lou (Robert Bosch Gmbh) &middot; Thomas Brox (University of Freiburg)</i></p>

      <p><b>The Point Where Reality Meets Fantasy: Mixed Adversarial Generators for Image Splice Detection</b><br><i>Vladimir V. Kniaz (IEEE) &middot; Vladimir Knyaz (State Research Institute of Aviation Systems) &middot; Fabio Remondino ("Fondazione Bruno Kessler, Italy")</i></p>

      <p><b>You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle</b><br><i>Dinghuai Zhang (Peking University) &middot; Tianyuan Zhang (Peking University) &middot; Yiping  Lu (Peking University) &middot; Zhanxing Zhu (Peking University) &middot; Bin Dong (Peking University)</i></p>

      <p><b>Imitation Learning from Observations by Minimizing Inverse Dynamics Disagreement</b><br><i>Chao Yang (Tsinghua University) &middot; Xiaojian Ma (University of California, Los Angeles) &middot; Wenbing Huang (Tsinghua University) &middot; Fuchun Sun (Tsinghua) &middot; 刘 华平 (清华大学) &middot; Junzhou Huang (University of Texas at Arlington / Tencent AI Lab) &middot; Chuang Gan (MIT-IBM Watson AI Lab)</i></p>

      <p><b>Asymptotic Guarantees for Learning Generative Models with the Sliced-Wasserstein Distance</b><br><i>Kimia Nadjahi ( Télécom ParisTech) &middot; Alain Durmus (ENS) &middot; Umut Simsekli (Institut Polytechnique de Paris) &middot; Roland Badeau (Télécom ParisTech)</i></p>

      <p><b>Generalized Sliced Wasserstein Distances</b><br><i>Soheil Kolouri (HRL Laboratories LLC) &middot; Kimia Nadjahi ( Télécom ParisTech) &middot; Umut Simsekli (Institut Polytechnique de Paris) &middot; Roland Badeau (Télécom ParisTech) &middot; Gustavo Rohde (University of Virginia)</i></p>

      <p><b>First Exit Time Analysis of Stochastic Gradient Descent Under Heavy-Tailed Gradient Noise</b><br><i>Than Huy Nguyen (Telecom ParisTech) &middot; Umut Simsekli (Institut Polytechnique de Paris) &middot; Mert Gurbuzbalaban (Rutgers) &middot; Gaël RICHARD (Télécom ParisTech)</i></p>

      <p><b>Blind Super-Resolution Kernel Estimation using an Internal-GAN</b><br><i>Yosef Bell Kligler (Weizmann Istitute of Science) &middot; Assaf Shocher (Weizmann Institute of Science) &middot; Michal Irani (The Weizmann Institute of Science)</i></p>

      <p><b>Noise-tolerant fair classification</b><br><i>Alex Lamy (Columbia University) &middot; Ziyuan Zhong (Columbia University) &middot; Aditya Menon (Google) &middot; Nakul Verma (Columbia University)</i></p>

      <p><b>Generalization in Generative Adversarial Networks: A Novel Perspective from Privacy Protection</b><br><i>Bingzhe Wu (Peeking University) &middot; Shiwan Zhao (IBM Research - China) &middot; Haoyang Xu (Peking University) &middot; Chaochao Chen (Ant Financial) &middot; Li Wang (Ant Financial) &middot; Xiaolu Zhang (Ant Financial Services Group) &middot; Guangyu Sun (Peking University) &middot; Jun Zhou (Ant Financial)</i></p>

      <p><b>Joint-task Self-supervised Learning for Temporal Correspondence</b><br><i>xueting li (uc merced) &middot; Sifei Liu (NVIDIA) &middot; Shalini De Mello (NVIDIA) &middot; Xiaolong Wang (CMU) &middot; Jan Kautz (NVIDIA) &middot; Ming-Hsuan Yang (UC Merced / Google)</i></p>

      <p><b>Provable Gradient Variance Guarantees for Black-Box Variational Inference</b><br><i>Justin Domke (University of Massachusetts, Amherst)</i></p>

      <p><b>Divide and Couple: Using Monte Carlo Variational Objectives for Posterior Approximation</b><br><i>Justin Domke (University of Massachusetts, Amherst) &middot; Daniel Sheldon (University of Massachusetts Amherst)</i></p>

      <p><b>Experience Replay for Continual Learning</b><br><i>David Rolnick (UPenn) &middot; Arun Ahuja (DeepMind) &middot; Jonathan Schwarz (DeepMind) &middot; Timothy Lillicrap (Google DeepMind) &middot; Gregory Wayne (Google DeepMind)</i></p>

      <p><b>Deep ReLU Networks Have Surprisingly Few Activation Patterns</b><br><i>Boris Hanin (Texas A&M) &middot; David Rolnick (UPenn)</i></p>

      <p><b>Chasing Ghosts: Instruction Following as Bayesian State Tracking</b><br><i>Peter Anderson (Georgia Tech) &middot; Ayush Shrivastava (Georgia Institute of Technology) &middot; Devi Parikh (Georgia Tech / Facebook AI Research (FAIR)) &middot; Dhruv Batra (Georgia Tech / Facebook AI Research (FAIR)) &middot; Stefan Lee (Georgia Institute of Technology)</i></p>

      <p><b>Block Coordinate Regularization by Denoising</b><br><i>Yu Sun (Washington University in St. Louis) &middot; Jiaming Liu (Washington University in St. Louis) &middot; Ulugbek Kamilov (Washington University in St. Louis)</i></p>

      <p><b>Reducing Noise in GAN Training with Variance Reduced Extragradient</b><br><i>Tatjana Chavdarova (Mila & Idiap & EPFL) &middot; Gauthier Gidel (Mila) &middot; François Fleuret (Idiap Research Institute) &middot; Simon Lacoste-Julien (Mila, Université de Montréal)</i></p>

      <p><b>Learning Erdos-Renyi Random Graphs via Edge Detecting Queries</b><br><i>Zihan Li (National University of Singapore) &middot; Matthias Fresacher (University of Adelaide) &middot; Jonathan Scarlett (National University of Singapore)</i></p>

      <p><b>A Primal-Dual link between GANs and Autoencoders</b><br><i>Hisham Husain (The Australian National University) &middot; Richard Nock (Data61, the Australian National University and the University of Sydney) &middot; Robert Williamson (Australian National University & Data61)</i></p>

      <p><b>muSSP: Efficient Min-cost Flow Algorithm for Multi-object Tracking</b><br><i>Congchao Wang (Virginia Tech) &middot; Yizhi Wang (Virginia Tech) &middot; Yinxue Wang (Virginia Tech) &middot; Chiung-Ting Wu (Virginia Tech) &middot; Guoqiang Yu (Virginia Tech)</i></p>

      <p><b>Category Anchor-Guided Unsupervised Domain Adaptation for Semantic Segmentation</b><br><i>Qiming Zhang (the University of Sydney) &middot; Jing Zhang (The University of Sydney) &middot; Wei Liu (Tencent AI Lab) &middot; Dacheng Tao (University of Sydney)</i></p>

      <p><b>Invert to Learn to Invert</b><br><i>Patrick Putzky (University of Amsterdam) &middot; Max Welling (University of Amsterdam / Qualcomm AI Research)</i></p>

      <p><b>Equitable Stable Matchings in Quadratic Time</b><br><i>Nikolaos Tziavelis (Northeastern University) &middot; Ioannis Giannakopoulos (National Technical University of Athens) &middot; Katerina Doka (NTUA) &middot; Nectarios Koziris (NTUA) &middot; Panagiotis Karras (Aarhus University)</i></p>

      <p><b>Zero-Shot Semantic Segmentation</b><br><i>Maxime Bucher (Valeo.ai) &middot; Tuan-Hung VU (Valeo.ai) &middot; Matthieu Cord (Sorbonne University) &middot; Patrick Pérez (Valeo.ai)</i></p>

      <p><b>Metric Learning for Adversarial Robustness</b><br><i>Chengzhi Mao (Columbia University) &middot; Ziyuan Zhong (Columbia University) &middot; Junfeng Yang (Columbia University) &middot; Carl Vondrick (Columbia University) &middot; Baishakhi Ray (Columbia University)</i></p>

      <p><b>DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction</b><br><i>Qiangeng Xu (USC) &middot; Weiyue Wang (USC) &middot; Duygu Ceylan (Adobe Research) &middot;  Radomir Mech (Adobe Systems Incorporated) &middot; Ulrich Neumann (USC)</i></p>

      <p><b>Batched Multi-armed Bandits Problem</b><br><i>Zijun Gao (Stanford University) &middot; Yanjun Han (Stanford University) &middot; Zhimei Ren (Stanford University) &middot; Zhengqing Zhou (Stanford University)</i></p>

      <p><b>vGraph: A Generative Model for Joint Community Detection and Node Representation Learning</b><br><i>Fan-Yun Sun (National Taiwan University) &middot; Meng Qu (MILA) &middot; Jordan Hoffmann (Harvard University/Mila) &middot; Chin-Wei Huang (MILA) &middot; Jian Tang (HEC Montreal & MILA)</i></p>

      <p><b>Differentially Private Bayesian Linear Regression</b><br><i>Garrett Bernstein (University of Massachusetts Amherst) &middot; Daniel Sheldon (University of Massachusetts Amherst)</i></p>

      <p><b>Semantic Conditioned Dynamic Modulation for Temporal Sentence Grounding in Videos</b><br><i>Yitian Yuan (Tsinghua University) &middot; Lin Ma (Tencent AI Lab) &middot; Jingwen Wang (Tencent AI Lab) &middot; Wei Liu (Tencent AI Lab) &middot; Wenwu Zhu (Tsinghua University)</i></p>

      <p><b>AGEM: Solving Linear Inverse Problems via Deep Priors and Sampling</b><br><i>Bichuan Guo (Tsinghua University) &middot; Yuxing Han (South China Agriculture University) &middot; Jiangtao Wen (Tsinghua University)</i></p>

      <p><b>CPM-Nets: Cross Partial Multi-View Networks</b><br><i>Changqing Zhang (Tianjin university) &middot; Zongbo Han (Tianjin University) &middot; yajie cui (tianjin university) &middot; Huazhu Fu (Inception Institute of Artificial Intelligence) &middot; Joey Tianyi Zhou (IHPC, A*STAR) &middot; Qinghua Hu (Tianjin University)</i></p>

      <p><b>Learning to Predict Layout-to-image Conditional Convolutions for Semantic Image Synthesis</b><br><i>Xihui Liu (The Chinese University of Hong Kong) &middot; Guojun Yin (University of Science and Technology of China) &middot; Jing Shao (Sensetime) &middot; Xiaogang Wang (The Chinese University of Hong Kong) &middot; hongsheng Li (cuhk)</i></p>

      <p><b>Staying up to Date with Online Content Changes Using Reinforcement Learning for Scheduling</b><br><i>Andrey Kolobov (Microsoft Research) &middot; Yuval Peres (N/A) &middot; Cheng Lu (Microsoft) &middot; Eric J Horvitz (Microsoft Research)</i></p>

      <p><b>SySCD: A System-Aware Parallel Coordinate Descent Algorithm</b><br><i>Celestine Mendler-Dünner (UC Berkeley) &middot; Nikolas Ioannou (IBM Research) &middot; Thomas Parnell (IBM Research)</i></p>

      <p><b>Importance Weighted Hierarchical Variational Inference</b><br><i>Artem Sobolev (Samsung) &middot; Dmitry Vetrov (Higher School of Economics, Samsung AI Center, Moscow)</i></p>

      <p><b>RSN: Randomized Subspace Newton</b><br><i>Robert Gower (Telecom-Paristech) &middot; Dmitry Koralev (KAUST) &middot; Felix Lieder (Heinrich-Heine-Universität Düsseldorf) &middot; Peter Richtarik (KAUST)</i></p>

      <p><b>Trust Region-Guided Proximal Policy Optimization</b><br><i>Yuhui Wang (Nanjing University of Aeronautics and Astronautics, China) &middot; Hao He (Nanjing University of Aeronautics and Astronautics) &middot; Xiaoyang Tan (Nanjing University of Aeronautics and Astronautics, China) &middot; Yaozhong Gan (Nanjing University of Aeronautics and Astronautics, China)</i></p>

      <p><b>Adversarial Self-Defense for Cycle-Consistent GANs</b><br><i>Dina Bashkirova (Boston University) &middot; Ben Usman (Boston University) &middot; Kate Saenko (Boston University)</i></p>

      <p><b>Towards closing the gap between the theory and practice of SVRG</b><br><i>Othmane Sebbouh (Télécom ParisTech) &middot; Nidham Gazagnadou (Télécom ParisTech) &middot; Samy Jelassi (Princeton University) &middot; Francis Bach (INRIA - Ecole Normale Superieure) &middot; Robert Gower (Telecom-Paristech)</i></p>

      <p><b>Uniform Error Bounds for Gaussian Process Regression with Application to Safe Control</b><br><i>Armin Lederer (Technical University of Munich) &middot; Jonas Umlauft (Technical University of Munich) &middot; Sandra Hirche (Technische Universitaet Muenchen)</i></p>

      <p><b>ETNet: Error Transition Network for Arbitrary Style Transfer</b><br><i>Chunjin Song (Shenzhen University) &middot; Zhijie Wu (Shenzhen University) &middot; Yang Zhou (Shenzhen University) &middot; Minglun Gong (Memorial Univ) &middot; Hui Huang (Shenzhen University)</i></p>

      <p><b>No Pressure! Addressing the Problem of Local Minima in Manifold Learning Algorithms</b><br><i>Max Vladymyrov (Google)</i></p>

      <p><b>Deep Equilibrium Models</b><br><i>Shaojie Bai (Carnegie Mellon University) &middot; J. Zico Kolter (Carnegie Mellon University / Bosch Center for AI) &middot; Vladlen Koltun (Intel Labs)</i></p>

      <p><b>Saccader: Accurate, Interpretable Image Classification with Hard Attention</b><br><i>Gamaleldin Elsayed (Google Brain) &middot; Simon Kornblith (Google Brain) &middot; Quoc V Le (Google)</i></p>

      <p><b>Multiway clustering via tensor block models  </b><br><i>Miaoyan Wang (University of Wisconsin - Madison) &middot; Yuchen Zeng (University of Wisconsin - Madison)</i></p>

      <p><b>Regret Minimization for Reinforcement Learning on Multi-Objective Online Markov Decision Processes</b><br><i>Wang Chi Cheung (Department of Industrial Systems Engineering and Management, National University of Singapore)</i></p>

      <p><b>NAT: Neural Architecture Transformer for Accurate and Compact Architectures</b><br><i>Yong Guo (South China University of Technology) &middot; Yin Zheng (Tencent AI Lab) &middot; Mingkui Tan (South China University of Technology) &middot; Qi Chen (South China University of Technology) &middot; Jian Chen ("South China University of Technology, China") &middot; Peilin Zhao (Tencent AI Lab) &middot; Junzhou Huang (University of Texas at Arlington / Tencent AI Lab)</i></p>

      <p><b>Selecting Optimal Decisions via Distributionally Robust Nearest-Neighbor Regression</b><br><i>Ruidi Chen (Boston University) &middot; Ioannis Paschalidis (Boston University)</i></p>

      <p><b>Network Pruning via Transformable Architecture Search</b><br><i>Xuanyi Dong (University of Technology Sydney) &middot; Yi Yang (UTS)</i></p>

      <p><b>Differentiable Cloth Simulation for Inverse Problems</b><br><i>Junbang Liang (University of Maryland, College Park) &middot; Ming Lin (UMD-CP & UNC-CH ) &middot; Vladlen Koltun (Intel Labs)</i></p>

      <p><b>Poisson-randomized Gamma Dynamical Systems</b><br><i>Aaron Schein (UMass Amherst) &middot; Scott Linderman (Columbia University) &middot; Mingyuan Zhou (University of Texas at Austin) &middot; David Blei (Columbia University) &middot; Hanna Wallach (MSR NYC)</i></p>

      <p><b>Volumetric Correspondence Networks for Optical Flow</b><br><i>Gengshan Yang (Carnegie Mellon University) &middot; Deva Ramanan (Carnegie Mellon University)</i></p>

      <p><b>Learning Conditional Deformable Templates with Convolutional Networks</b><br><i>Adrian Dalca (MIT, HMS) &middot; Marianne Rakic (MIT/ETH Zürich) &middot; John Guttag (Massachusetts Institute of Technology) &middot; Mert Sabuncu (Cornell)</i></p>

      <p><b>Fast Low-rank Metric Learning for Large-scale and High-dimensional Data</b><br><i>Han Liu (Tsinghua University) &middot; Zhizhong Han (University of Maryland, College Park) &middot; Yu-Shen Liu (Tsinghua University) &middot; Ming Gu (Tsinghua University)</i></p>

      <p><b>Efficient Symmetric Norm Regression via Linear Sketching</b><br><i>Zhao Song (University of Washington) &middot; Ruosong Wang (Carnegie Mellon University) &middot; Lin Yang (Johns Hopkins University) &middot; Hongyang Zhang (Carnegie Mellon University) &middot; Peilin Zhong (Columbia University)</i></p>

      <p><b>RUBi: Reducing Unimodal Biases in Visual Question Answering</b><br><i>Remi Cadene (LIP6) &middot; Corentin Dancette (LIP6) &middot; Hedi Ben younes (Université Pierre & Marie Curie / Heuritech) &middot; Matthieu Cord (Sorbonne University) &middot; Devi Parikh (Georgia Tech / Facebook AI Research (FAIR))</i></p>

      <p><b>Reducing Scene Bias of Convolutional Neural Networks for Human Action Understanding</b><br><i>Jinwoo Choi (Virginia Tech) &middot; Chen Gao (Virginia Tech) &middot; Joseph C.E. Messou (Virginia Tech) &middot; Jia-Bin Huang (Virginia Tech)</i></p>

      <p><b>NeurVPS: Neural Vanishing Point Scanning via Conic Convolution</b><br><i>Yichao Zhou (UC Berkeley) &middot; Haozhi Qi (UC Berkeley) &middot; Jingwei Huang (Stanford University) &middot; Yi Ma (UC Berkeley)</i></p>

      <p><b>DATA: Differentiable ArchiTecture Approximation</b><br><i>Jianlong Chang (National Laboratory of Pattern Recognition, Institute of Automation, Chinese Academy of Sciences) &middot; xinbang zhang (Institute of Automation,Chinese  Academy of Science) &middot; Yiwen Guo (Intel Labs China) &middot; GAOFENG MENG (Institute of Automation, Chinese Academy of Sciences) &middot; SHIMING XIANG (Chinese Academy of Sciences, China) &middot; Chunhong Pan (Institute of Automation, Chinese Academy of Sciences)</i></p>

      <p><b>Learn, Imagine and Create: Text-to-Image Generation from Prior Knowledge</b><br><i>Tingting Qiao (Zhejiang University) &middot; Jing Zhang (The University of Sydney) &middot; Duanqing Xu (Zhejiang University) &middot; Dacheng Tao (University of Sydney)</i></p>

      <p><b>Memory-oriented Decoder for Light Field Salient Object Detection</b><br><i>Miao Zhang (Dalian University of Technology) &middot; Jingjing Li (Dalian University of Technology) &middot; Wei Ji (Dalian University of Technology) &middot; Yongri Piao (Dalian University of Technology) &middot; Huchuan Lu (Dalian University of Technology)</i></p>

      <p><b>Multi-label Co-regularization for Semi-supervised Facial Action Unit Recognition</b><br><i>Xuesong Niu (Institute of Computing Technology, CAS) &middot; Hu Han (ICT, CAS) &middot; Shiguang Shan (Chinese Academy of Sciences) &middot; Xilin Chen (Institute of Computing Technology, Chinese Academy of Sciences)</i></p>

      <p><b>Correlated Uncertainty for Learning Dense Correspondences from Noisy Labels</b><br><i>Natalia Neverova (Facebook AI Research) &middot; David Novotny (Facebook AI Research) &middot; Andrea Vedaldi (University of Oxford / Facebook AI Research)</i></p>

      <p><b>Powerset Convolutional Neural Networks</b><br><i>Chris Wendler (ETH Zurich) &middot; Markus Püschel (ETH Zurich) &middot; Dan Alistarh (IST Austria)</i></p>

      <p><b>Optimal Pricing in Repeated Posted-Price Auctions with Different Patience of the Seller and the Buyer</b><br><i>Arsenii Vanunts (Yandex) &middot; Alexey Drutsa (Yandex)</i></p>

      <p><b>An Accelerated Decentralized Stochastic Proximal Algorithm for Finite Sums</b><br><i>Hadrien Hendrikx (INRIA) &middot; Francis Bach (INRIA - Ecole Normale Superieure) &middot; Laurent Massoulié (Inria)</i></p>

      <p><b>Efficient 3D Deep Learning via Point-Based Representation and Voxel-Based Convolution</b><br><i>Zhijian Liu (MIT) &middot; Haotian Tang (Shanghai Jiao Tong University) &middot; Yujun Lin (MIT) &middot; Song Han (MIT)</i></p>

      <p><b>Deep Learning without Weight Transport</b><br><i>Mohamed Akrout (University of Toronto) &middot; Collin Wilson (University of Toronto) &middot; Peter Humphreys (Google) &middot; Timothy Lillicrap (Google DeepMind) &middot; Douglas Tweed (University of Toronto)</i></p>

      <p><b>Combinatorial Bandits with Relative Feedback </b><br><i>Aadirupa Saha (Indian Institute of SCience) &middot; Aditya Gopalan (Indian Institute of Science)</i></p>

      <p><b>General Proximal Incremental Aggregated Gradient Algorithms: Better and Novel Results under General Scheme</b><br><i>Tao Sun (National university of defense technology) &middot; Yuejiao Sun (University of California, Los Angeles) &middot; Dongsheng Li (School of Computer Science, National University of Defense Technology) &middot; Qing Liao (Harbin Institute of Technology (Shenzhen))</i></p>

      <p><b>Joint Optimizing of Cycle-Consistent Networks</b><br><i>Leonidas J Guibas (stanford.edu) &middot; Qixing Huang (The University of Texas at Austin) &middot; Zhenxiao Liang (The University of Texas at Austin)</i></p>

      <p><b>Explicit Disentanglement of Appearance and Perspective in Generative Models</b><br><i>Nicki Skafte Detlefsen (Technical University of Denmark) &middot; Søren Hauberg (Technical University of Denmark)</i></p>

      <p><b>Polynomial Cost of Adaptation for X-Armed Bandits</b><br><i>Hedi Hadiji (Laboratoire de Mathematiques d’Orsay, Univ. Paris-Sud,)</i></p>

      <p><b>Learning to Propagate for Graph Meta-Learning</b><br><i>LU LIU (University of Technology Sydney) &middot; Tianyi Zhou (University of Washington, Seattle) &middot; Guodong Long (University of Technology Sydney) &middot; Jing Jiang (University of Technology Sydney) &middot; Chengqi Zhang (University of Technology Sydney)</i></p>

      <p><b>Secretary Ranking with Minimal Inversions</b><br><i>Sepehr Assadi (Princeton University) &middot; Eric Balkanski (Harvard University) &middot; Renato Leme (Google Research)</i></p>

      <p><b>Nonparametric Regressive Point Processes Based on Conditional Gaussian Processes</b><br><i>Siqi Liu (University of Pittsburgh) &middot; Milos Hauskrecht (University of Pittsburgh)</i></p>

      <p><b>Learning Perceptual Inference by Contrasting</b><br><i>Chi  Zhang (University of California, Los Angeles) &middot; Baoxiong Jia (UCLA) &middot; Feng Gao (UCLA) &middot; Yixin Zhu (University of California, Los Angeles) &middot; HongJing Lu (UCLA) &middot; Song-Chun Zhu (UCLA)</i></p>

      <p><b>Selecting the independent coordinates of manifolds with large aspect ratios</b><br><i>Yu-Chia Chen (University of Washington) &middot; Marina Meila (University of Washington)</i></p>

      <p><b>Region-specific Diffeomorphic Metric Mapping</b><br><i>Zhengyang Shen (University of North Carolina at Chapel Hill) &middot; Francois-Xavier Vialard (University Paris-Est) &middot; Marc Niethammer (UNC Chapel Hill)</i></p>

      <p><b>Subset Selection via Supervised Facility Location</b><br><i>Chengguang Xu (Northeastern University) &middot; Ehsan Elhamifar (Northeastern University)</i></p>

      <p><b>Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations</b><br><i>Vincent Sitzmann (Stanford University) &middot; Michael Zollhoefer (Stanford University) &middot; Gordon Wetzstein (Stanford University)</i></p>

      <p><b>Reconciling λ-Returns with Experience Replay</b><br><i>Brett Daley (Northeastern University) &middot; Christopher Amato (Northeastern University)</i></p>

      <p><b>Control Batch Size and Learning Rate to Generalize Well: Theoretical and Empirical Evidence</b><br><i>Fengxiang He (The University of Sydney) &middot; Tongliang Liu (The University of Sydney) &middot; Dacheng Tao (University of Sydney)</i></p>

      <p><b>Non-Asymptotic Gap-Dependent Regret Bounds for Tabular MDPs</b><br><i>Max Simchowitz (Berkeley) &middot; Kevin Jamieson (U Washington)</i></p>

      <p><b>A Graph Theoretic Framework of Recomputation Algorithms for Memory-Efficient Backpropagation</b><br><i>Mitsuru Kusumoto (Preferred Networks, Inc.) &middot; Takuya Inoue (University of Tokyo) &middot; Gentaro Watanabe (Preferred Networks, Inc.) &middot; Takuya Akiba (Preferred Networks, Inc.) &middot; Masanori Koyama (Preferred Networks Inc. )</i></p>

      <p><b>Combinatorial Inference against Label Noise</b><br><i>Paul Hongsuck Seo (POSTECH) &middot; Geeho Kim (Seoul National University) &middot; Bohyung Han (Seoul National University)</i></p>

      <p><b> Value Propagation for Decentralized Networked Deep Multi-agent  Reinforcement Learning</b><br><i>Chao Qu (Ant Financial Services Group) &middot; Shie Mannor (Technion) &middot; Huan Xu (Georgia Inst. of Technology) &middot; Yuan Qi (Ant Financial Services Group) &middot; Le Song (Ant Financial Services Group) &middot; Junwu Xiong (Ant Financial Services Group)</i></p>

      <p><b>Convolution with even-sized kernels and symmetric padding</b><br><i>Shuang Wu (Tsinghua University) &middot; Guanrui Wang (Tsinghua University) &middot; Pei Tang (Tsinghua University) &middot; Feng Chen (Tsinghua University) &middot; Luping Shi (tsinghua university)</i></p>

      <p><b>On The Classification-Distortion-Perception Tradeoff</b><br><i>Dong Liu (University of Science and Technology of China) &middot; Haochen Zhang (University of Science and Technology of China) &middot; Zhiwei Xiong (University of Science and Technology of China)</i></p>

      <p><b>Optimal Statistical Rates for Decentralised Non-Parametric Regression with Linear Speed-Up</b><br><i>Dominic Richards (University of Oxford) &middot; Patrick Rebeschini (University of Oxford)</i></p>

      <p><b>Online sampling from log-concave distributions</b><br><i>Holden Lee (Princeton University) &middot; Oren Mangoubi (EPFL) &middot; Nisheeth Vishnoi (Yale University)</i></p>

      <p><b>Envy-Free Classification</b><br><i>Maria-Florina Balcan (Carnegie Mellon University) &middot; Travis Dick (Carnegie Mellon University) &middot; Ritesh Noothigattu (Carnegie Mellon University) &middot; Ariel D Procaccia (Carnegie Mellon University)</i></p>

      <p><b>Finding Friend and Foe in Multi-Agent Games</b><br><i>Jack S Serrino (MIT) &middot; Max Kleiman-Weiner (Harvard) &middot; David Parkes (Harvard University) &middot; Josh Tenenbaum (MIT)</i></p>

      <p><b>Computer Vision with a Single (Robust) Classifier</b><br><i>Shibani Santurkar (MIT) &middot; Andrew Ilyas (MIT) &middot; Dimitris Tsipras (MIT) &middot; Logan Engstrom (MIT) &middot; Brandon Tran (Massachusetts Institute of Technology) &middot; Aleksander Madry (MIT)</i></p>

      <p><b>Gated CRF Loss for Weakly Supervised Semantic Image Segmentation</b><br><i>Anton Obukhov (ETH Zurich) &middot; Stamatios Georgoulis (ETH Zurich) &middot; Dengxin Dai (ETH Zurich) &middot; Luc V Gool (Computer Vision Lab, ETH Zurich)</i></p>

      <p><b>Model Compression with Adversarial Robustness: A Unified Optimization Framework</b><br><i>Shupeng Gui (University of Rochester) &middot; Haotao N Wang (Texas A&M University) &middot; Haichuan Yang (University of Rochester) &middot; Chen Yu (University of Rochester) &middot; Zhangyang Wang (TAMU) &middot; Ji Liu (University of Rochester, Tencent AI lab)</i></p>

      <p><b>Neuron Communication Networks</b><br><i>Jianwei Yang (Georgia Tech) &middot; Zhile Ren (Georgia Tech) &middot; Chuang Gan (MIT-IBM Watson AI Lab) &middot; Hongyuan Zhu (Astar) &middot; Ji Lin (MIT) &middot; Devi Parikh (Georgia Tech / Facebook AI Research (FAIR))</i></p>

      <p><b>CondConv: Conditionally Parameterized Convolutions for Efficient Inference</b><br><i>Brandon Yang (Google Brain) &middot; Gabriel Bender (Google Brain) &middot; Quoc V Le (Google) &middot; Jiquan Ngiam (Google Brain)</i></p>

      <p><b>Regression Planning Networks</b><br><i>Danfei Xu (Stanford University) &middot; Roberto Martín-Martín (Stanford University) &middot; De-An Huang (Stanford University) &middot; Yuke Zhu (Stanford University) &middot; Silvio Savarese (Stanford University) &middot; Li Fei-Fei (Stanford University)</i></p>

      <p><b>Twin Auxilary Classifiers GAN</b><br><i>Mingming Gong (University of Melbourne) &middot; Yanwu Xu (University of Pittsburgh) &middot; Chunyuan Li (Microsoft Research) &middot; Kun Zhang (CMU) &middot; Kayhan Batmanghelich (University of Pittsburgh)</i></p>

      <p><b>Conditional Structure Generation through Graph Variational Generative Adversarial Nets</b><br><i>Carl Yang (University of Illinois, Urbana Champaign) &middot; Peiye Zhuang (UIUC) &middot; Wenhan Shi (UIUC) &middot; Alan Luu (UIUC) &middot; Pan Li (Stanford)</i></p>

      <p><b>Distributional Policy Optimization: An Alternative Approach for Continuous Control</b><br><i>Chen Tessler (Technion) &middot; Guy Tennenholtz (Technion) &middot; Shie Mannor (Technion)</i></p>

      <p><b>Sampling Sketches for Concave Sublinear Functions of Frequencies</b><br><i>Edith Cohen (Google) &middot; Ofir Geri (Stanford University)</i></p>

      <p><b>Deliberative Explanations: visualizing network insecurities</b><br><i>Pei Wang (UC San Diego) &middot; Nuno Nvasconcelos (UC San Diego)</i></p>

      <p><b>Computing Full Conformal Prediction Set with Approximate Homotopy</b><br><i>Eugene Ndiaye (Riken AIP) &middot; Ichiro Takeuchi (Nagoya Institute of Technology)</i></p>

      <p><b>Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift</b><br><i>Stephan Rabanser (Amazon) &middot; Stephan Günnemann (Technical University of Munich) &middot; Zachary Lipton (Carnegie Mellon University)</i></p>

      <p><b>Hierarchical Reinforcement Learning with Advantage-Based Auxiliary Rewards</b><br><i>Siyuan Li (Tsinghua University) &middot; Rui Wang (Tsinghua University) &middot; Minxue Tang (Tsinghua University) &middot; Chongjie Zhang (Tsinghua University)</i></p>

      <p><b>Multi-View Reinforcement Learning</b><br><i>Minne Li (University College London) &middot; Lisheng Wu (UCL) &middot; Jun WANG (UCL)</i></p>

      <p><b>Cascade RPN: Delving into High-Quality Region Proposal Network with Adaptive Convolution</b><br><i>Thang Vu (KAIST) &middot; Hyunjun Jang (KAIST) &middot; Trung Pham (KAIST) &middot; Chang Yoo (KAIST)</i></p>

      <p><b>Neural Diffusion Distance for Image Segmentation</b><br><i>Jian Sun (Xi'an Jiaotong University) &middot; Zongben Xu (XJTU)</i></p>

      <p><b>Fine-grained Optimization of Deep Neural Networks</b><br><i>Mete Ozay (Independent Researcher (N/A))</i></p>

      <p><b>Extending Stein’s Unbiased Risk Estimator To Train Deep Denoisers with Correlated Pairs of Noisy Images</b><br><i>Magauiya Zhussip (UNIST) &middot; Shakarim Soltanayev (Ulsan National Institute of Science and Technology) &middot; Se Young Chun (UNIST)</i></p>

      <p><b>Wibergian Learning of Continuous Energy Functions</b><br><i>Chris Russell (The Alan Turing Institute/ The University of Surrey) &middot; Matteo Toso (University of Surrey) &middot; Neill Campbell (University of Bath)</i></p>

      <p><b>Hyperspherical Prototype Networks</b><br><i>Pascal Mettes (University of Amsterdam) &middot; Elise van der Pol (University of Amsterdam) &middot; Cees Snoek (University of Amsterdam)</i></p>

      <p><b>Expressive power of tensor-network factorizations for probabilistic modelling</b><br><i>Ivan Glasser (Max Planck Institute of Quantum Optics) &middot; Ryan Sweke (Freie Universitaet Berlin) &middot; Nicola Pancotti (Max Planck Institute of Quantum Optics) &middot; Jens Eisert (Freie Universitaet Berlin) &middot; Ignacio Cirac (Max-Planck Institute of Quantum Optics)</i></p>

      <p><b>HyperGCN: A New Method For Training Graph Convolutional Networks on Hypergraphs</b><br><i>Naganand Yadati (Indian Institute of Science) &middot; Madhav Nimishakavi (Indian Institute of Science) &middot; Prateek Yadav (Indian Institute of Science) &middot; Vikram Nitin (Indian Institute of Science) &middot; Anand Louis (Indian Institute of Science, Bangalore, India) &middot; Partha Talukdar (Indian Institute of Science, Bangalore)</i></p>

      <p><b>SSRGD: Simple Stochastic Recursive Gradient Descent for Escaping Saddle Points</b><br><i>Zhize Li (Tsinghua University)</i></p>

      <p><b>Efficient Meta Learning via Minibatch Proximal Update</b><br><i>Pan Zhou (National University of Singapore) &middot; Xiaotong Yuan (Nanjing University of Information Science & Technology) &middot; Huan Xu (Alibaba Group) &middot; Shuicheng Yan (National University of Singapore) &middot; Jiashi Feng (National University of Singapore)</i></p>

      <p><b>Unconstrained Monotonic Neural Networks</b><br><i>Antoine Wehenkel (ULiège) &middot; Gilles Louppe (University of Liège)</i></p>

      <p><b>Guided Similarity Separation for Image Retrieval</b><br><i>Chundi Liu (Layer6 AI) &middot; Guangwei Yu (Layer6) &middot; Maksims Volkovs (layer6.ai) &middot; Cheng Chang (Layer6 AI) &middot; Himanshu Rai (Layer6 AI) &middot; Junwei Ma (Layer6 AI) &middot; Satya Krishna Gorti (Layer6 AI)</i></p>

      <p><b>Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss</b><br><i>Kaidi Cao (Stanford University) &middot; Colin Wei (Stanford University) &middot; Adrien Gaidon (Toyota Research Institute) &middot; Nikos Arechiga (Toyota Research Institute) &middot; Tengyu Ma (Stanford)</i></p>

      <p><b>Strategizing against No-regret Learners</b><br><i>Yuan Deng (Duke University) &middot; Jon Schneider (Google Research) &middot; Balasubramanian Sivan (Google Research)</i></p>

      <p><b>D-VAE: A Variational Autoencoder for Directed Acyclic Graphs</b><br><i>Muhan Zhang (Washington University in St. Louis) &middot; Shali Jiang (Washington University in St. Louis) &middot; Zhicheng Cui (Washington University in St. Louis) &middot; Roman Garnett (Washington University in St. Louis) &middot; Yixin Chen (Washington University in St. Louis)</i></p>

      <p><b>Hierarchical Optimal Transport for Document Representation</b><br><i>Mikhail Yurochkin (IBM Research, MIT-IBM Watson AI Lab) &middot; Sebastian Claici (MIT) &middot; Edward Chien (Massachusetts Institute of Technology) &middot; Farzaneh Mirzazadeh (IBM Research, MIT-IBM Watson AI Lab) &middot; Justin M Solomon (MIT)</i></p>

      <p><b>Multivariate Sparse Coding of Nonstationary Covariances with Gaussian Processes</b><br><i>Rui Li (Rochester Institute of Technology)</i></p>

      <p><b>Positional Normalization</b><br><i>Boyi Li (Cornell University) &middot; Felix Wu (Cornell University) &middot; Kilian Weinberger (Cornell University) &middot; Serge Belongie (Cornell University)</i></p>

      <p><b>A New Defense Against Adversarial Images: Turning a Weakness into a Strength</b><br><i>Shengyuan Hu (Cornell University) &middot; Tao Yu (Cornell University) &middot; Chuan Guo (Cornell University) &middot; Wei-Lun Chao (Cornell University          Ohio State University (OSU)) &middot; Kilian Weinberger (Cornell University)</i></p>

      <p><b>Quadratic Video Interpolation</b><br><i>Xiangyu Xu (Tsinghua University) &middot; Li Si-Yao (Beijing Normal University) &middot; Wenxiu Sun (SenseTime Research) &middot; Qian Yin (Beijing Normal University) &middot; Ming-Hsuan Yang (UC Merced / Google)</i></p>

      <p><b>ResNets Ensemble via the Feynman-Kac Formalism to Improve Natural and Robust Accuracies</b><br><i>Bao Wang (UCLA) &middot; Zuoqiang  Shi (zqshi@mail.tsinghua.edu.cn) &middot; Stanley Osher (UCLA)</i></p>

      <p><b>Incremental Scene Synthesis</b><br><i>Benjamin Planche (Siemens Corporate Technology) &middot; Xuejian Rong (City University of New York) &middot; Ziyan Wu (Siemens Corporation) &middot; Srikrishna Karanam (Siemens Corporate Technology, Princeton) &middot; Harald Kosch (PASSAU) &middot; YingLi Tian (City University of New York) &middot; Jan Ernst (Siemens Research) &middot; ANDREAS HUTTER (Siemens Corporate Technology, Germany)</i></p>

      <p><b>Self-Supervised Generalisation with Meta Auxiliary Learning</b><br><i>Shikun Liu (Imperial College London) &middot; Andrew Davison (Imperial College London) &middot; Edward Johns (Imperial College London)</i></p>

      <p><b>Variational Denoising Network: Toward Blind Noise Modeling and Removal</b><br><i>Zongsheng Yue (Xi'an Jiaotong University) &middot; Hongwei Yong (The Hong Kong Polytechnic University) &middot; Qian Zhao (Xi'an Jiaotong University) &middot; Deyu Meng (Xi'an Jiaotong University) &middot; Lei Zhang (The Hong Kong Polytechnic Univ)</i></p>

      <p><b>Fast Sparse Group Lasso</b><br><i>Yasutoshi Ida (NTT) &middot; Yasuhiro Fujiwara (NTT Software Innovation Center) &middot; Hisashi Kashima (Kyoto University/RIKEN Center for AIP)</i></p>

      <p><b>Learnable Tree Filter for Structure-preserving Feature Transform</b><br><i>Lin Song (Xi'an Jiaotong University) &middot; Yanwei Li (Institute of Automation, Chinese Academy of Sciences) &middot; Zeming Li (Megvii(Face++) Inc) &middot; Gang Yu (Megvii Inc) &middot; Hongbin Sun (Xi'an Jiaotong University) &middot; Jian Sun (Megvii, Face++) &middot; Nanning Zheng (Xi'an Jiaotong University)</i></p>

      <p><b>Data-Dependence of Plateau Phenomenon in Learning with Neural Network --- Statistical Mechanical Analysis</b><br><i>Yuki Yoshida (The University of Tokyo) &middot; Masato Okada (The University of Tokyo)</i></p>

      <p><b>Coordinated hippocampal-entorhinal replay as structural inference</b><br><i>Talfan Evans (University College London) &middot; Neil Burgess (University College London)</i></p>

      <p><b>Cascaded Dilated Dense Network with Two-step Data Consistency for MRI Reconstruction</b><br><i>Hao Zheng (East China Normal University) &middot; Faming Fang (East China Normal University) &middot; Guixu Zhang (East China Normal University)</i></p>

      <p><b>On the Ineffectiveness of Variance Reduced Optimization for Deep Learning</b><br><i>Aaron Defazio (Facebook AI Research) &middot; Leon Bottou (FAIR)</i></p>

      <p><b>On the Curved Geometry of Accelerated Optimization</b><br><i>Aaron Defazio (Facebook AI Research)</i></p>

      <p><b>Multi-marginal Wasserstein GAN</b><br><i>Jiezhang Cao (South China University of Technology) &middot; Langyuan Mo (South China University of Technology) &middot; Yifan   Zhang (South China University of Technology) &middot; Kui Jia (South China University of Technology) &middot; Chunhua Shen (University of Adelaide) &middot; Mingkui Tan (South China University of Technology)</i></p>

      <p><b>Better Exploration with Optimistic Actor Critic</b><br><i>Kamil Ciosek (Microsoft) &middot; Quan Vuong (University of California San Diego) &middot; Robert Loftin (Microsoft Research) &middot; Katja Hofmann (Microsoft Research)</i></p>

      <p><b>Importance Resampling for Off-policy Prediction</b><br><i>Matthew Schlegel (University of Alberta) &middot; Wesley Chung (University of Alberta) &middot; Daniel Graves (Huawei) &middot; Jian Qian (University of Alberta) &middot; Martha White (University of Alberta)</i></p>

      <p><b>The Label Complexity of Active Learning from Observational Data</b><br><i>Songbai Yan (University of California, San Diego) &middot; Kamalika Chaudhuri (UCSD) &middot; Tara Javidi (University of California San Diego)</i></p>

      <p><b>Meta-Learning Representations for Continual Learning</b><br><i>Khurram Javed (University of Alberta) &middot; Martha White (University of Alberta)</i></p>

      <p><b>Defense Against Adversarial Attacks Using Feature Scattering-based Adversarial Training</b><br><i>Haichao Zhang (Horizon Robotics) &middot; Jianyu Wang (Baidu USA)</i></p>

      <p><b>Visualizing the PHATE of Neural Networks</b><br><i>Scott Gigante (Yale University) &middot; Adam S Charles (Princeton University) &middot; Smita Krishnaswamy (Yale University) &middot; Gal Mishne (Yale)</i></p>

      <p><b>The Cells Out of Sample (COOS) dataset and benchmarks for measuring out-of-sample generalization of image classifiers</b><br><i>Alex X Lu (University of Toronto) &middot; Amy X Lu (University of Toronto/Vector Institute) &middot; Wiebke Schormann (Sunnybrook Research Institute) &middot; David Andrews (Sunnybrook Research Institute) &middot; Alan Moses (University of Toronto)</i></p>

      <p><b>Nonconvex Low-Rank Tensor Completion from Noisy Data</b><br><i>Changxiao Cai (Princeton University) &middot; Gen Li (Tsinghua University) &middot; H. Vincent Poor (Princeton University) &middot; Yuxin Chen (Princeton University)</i></p>

      <p><b>Beyond Online Balanced Descent: An Optimal Algorithm for Smoothed Online Optimization</b><br><i>Gautam Goel (Caltech) &middot; Yiheng Lin (Institute for Interdisciplinary Information Sciences, Tsinghua University) &middot; Haoyuan Sun (California Institute of Technology) &middot; Adam Wierman (California Institute of Technology)</i></p>

      <p><b>Channel Gating Neural Networks</b><br><i>Weizhe Hua (Cornell University) &middot; Yuan Zhou (Cornell) &middot; Christopher De Sa (Cornell) &middot; Zhiru Zhang (Cornell Univeristy) &middot; G. Edward Suh (Cornell University)</i></p>

      <p><b>Neural networks grown and self-organized by noise</b><br><i>Guruprasad Raghavan (California Institute of Technology) &middot; Matt Thomson (California Institute of Technology)</i></p>

      <p><b>Catastrophic Forgetting Meets Negative Transfer: Batch Spectral Shrinkage for Safe Transfer Learning</b><br><i>Xinyang Chen (Tsinghua University) &middot; Sinan Wang (Tsinghua University) &middot; Bo Fu (Tsinghua University) &middot; Mingsheng Long (Tsinghua University) &middot; Jianmin Wang (Tsinghua University)</i></p>

      <p><b>Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting</b><br><i>Jun Shu (Xi'an Jiaotong University) &middot; Qi Xie (Xi'an Jiaotong University) &middot; Lixuan Yi (Xi'an Jiaotong University) &middot; Qian Zhao (Xi'an Jiaotong University) &middot; Sanping Zhou (Xi'an Jiaotong University) &middot; Zongben Xu (Xi'an Jiaotong University) &middot; Deyu Meng (Xi'an Jiaotong University)</i></p>

      <p><b>Variational Structured Semantic Inference for Diverse Image Captioning</b><br><i>Fuhai Chen (Xiamen University) &middot; Rongrong Ji (Xiamen University, China) &middot; Jiayi Ji (Xiamen University) &middot; Xiaoshuai Sun (Xiamen University) &middot; Baochang Zhang (Beihang University) &middot; Xuri Ge (Xiamen University) &middot; Yongjian Wu (Tencent Technology (Shanghai) Co.,Ltd) &middot; Feiyue Huang (Tencent) &middot; Yan Wang (Microsoft)</i></p>

      <p><b>Mapping State Space using Landmarks for Universal Goal Reaching</b><br><i>Zhiao Huang (University of California San Diego) &middot; Hao Su (University of California San Diego) &middot; Fangchen Liu (UCSD)</i></p>

      <p><b>Transferable Normalization: Towards Improving Transferability of Deep Neural Networks</b><br><i>Ximei Wang (Tsinghua University) &middot; Ying Jin (Tsinghua University) &middot; Mingsheng Long (Tsinghua University) &middot; Jianmin Wang (Tsinghua University) &middot; Michael Jordan (UC Berkeley)</i></p>

      <p><b>Random deep neural networks are biased towards simple functions</b><br><i>Giacomo De Palma (Massachusetts Institute of Technology) &middot; Bobak Kiani (Massachusetts Institute of Technology) &middot; Seth Lloyd (MIT)</i></p>

      <p><b>XNAS: Neural Architecture Search with Expert Advice</b><br><i>Niv Nayman (Alibaba Group) &middot; Asaf Noy (Alibaba) &middot; Tal Ridnik (MIIL Alibaba) &middot; Itamar Friedman (Alibaba) &middot; Jing Rong (Alibaba) &middot; Lihi Zelnik (Alibaba)</i></p>

      <p><b>CNN^{2}: Viewpoint Generalization via a Binocular Vision</b><br><i>Wei-Da Chen (National Tsing Hua University) &middot; Shan-Hung Wu (National Tsing Hua University)</i></p>

      <p><b> Generalized Off-Policy Actor-Critic</b><br><i>Shangtong Zhang (University of Oxford) &middot; Wendelin Boehmer (University of Oxford) &middot; Shimon Whiteson (University of Oxford)</i></p>

      <p><b>DAC: The Double Actor-Critic Architecture for Learning Options</b><br><i>Shangtong Zhang (University of Oxford) &middot; Shimon Whiteson (University of Oxford)</i></p>

      <p><b>Numerically Accurate Hyperbolic Embeddings Using Tiling-Based Models</b><br><i>Tao Yu (Cornell University) &middot; Christopher De Sa (Cornell)</i></p>

      <p><b>Controlling Neural Level Sets</b><br><i>Matan Atzmon (Weizmann Institute Of Science) &middot; Niv Haim (Weizmann Institute of Science) &middot; Lior Yariv (Weizmann Institute of Science) &middot; Ofer Israelov (Weizmann Institute of Science) &middot; Haggai Maron (Weizmann Institute, Israel) &middot; Yaron Lipman (Weizmann Institute of Science)</i></p>

      <p><b>Blended Matching Pursuit</b><br><i>Cyrille Combettes (Georgia Institute of Technology) &middot; Sebastian Pokutta (Georgia Institute of Technology)</i></p>

      <p><b>An Improved Analysis of Training Over-parameterized Deep Neural Networks</b><br><i>Difan Zou (University of California, Los Angeles) &middot; Quanquan Gu (UCLA)</i></p>

      <p><b>Controllable Text to Image Generation</b><br><i>Bowen Li (University of Oxford) &middot; Xiaojuan Qi (University of Oxford) &middot; Thomas Lukasiewicz (University of Oxford) &middot; Philip Torr (University of Oxford)</i></p>

      <p><b>Improving Textual Network Learning with Variational Homophilic Embeddings</b><br><i>Wenlin Wang (Duke Univeristy) &middot; Chenyang Tao (Duke University) &middot; Zhe Gan (Microsoft) &middot; Guoyin Wang (Duke University) &middot; Liqun Chen (Duke University) &middot; Xinyuan Zhang (Duke University) &middot; Ruiyi Zhang (Duke University) &middot; Qian Yang (Duke University) &middot; Ricardo Henao (Duke University) &middot; Lawrence Carin (Duke University)</i></p>

      <p><b>Rethinking Generative Coverage: A Pointwise Guaranteed Approach</b><br><i>Peilin Zhong (Columbia University) &middot; Yuchen Mo (Columbia University) &middot; Chang Xiao (Columbia University) &middot; Pengyu Chen (Columbia University) &middot; Changxi Zheng (Columbia University)</i></p>

      <p><b>The Randomized Midpoint Method for Log-Concave Sampling</b><br><i>Ruoqi Shen (University of Washington) &middot; Yin Tat Lee (UW)</i></p>

      <p><b>Sample-Efficient Deep Reinforcement Learning via Episodic Backward Update</b><br><i>Su Young Lee (KAIST) &middot; Choi Sungik (KAIST) &middot; Sae-Young Chung (KAIST)</i></p>

      <p><b>Fully Neural Network based Model for General Temporal Point Processes</b><br><i>Takahiro Omi (The University of Tokyo) &middot; naonori ueda (RIKEN AIP) &middot; Kazuyuki Aihara (The University of Tokyo)</i></p>

      <p><b>Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks</b><br><i>Zhonghui You (Peking University) &middot; Kun Yan (Peking University) &middot; Jinmian Ye (SMILE Lab) &middot; Meng Ma (Peking University) &middot; Ping Wang (Peking University)</i></p>

      <p><b>Discrimination in Online Markets: Effects of Social Bias on Learning from Reviews and Policy Design</b><br><i>Faidra Monachou (Stanford University) &middot; Itai Ashlagi (Stanford)</i></p>

      <p><b>Provably Powerful Graph Networks</b><br><i>Haggai Maron (Weizmann Institute, Israel) &middot; Heli Ben-Hamu (Weizmann Institute of Science) &middot; Hadar Serviansky (WEIZMANN INSTITUTE OF SCIENCE) &middot; Yaron Lipman (Weizmann Institute of Science)</i></p>

      <p><b>Order Optimal One-Shot Distributed Learning</b><br><i>Arsalan Sharifnassab (Sharif University of Technology) &middot; Saber Salehkaleybar (Sharif University of Technology) &middot; S. Jamaloddin Golestani (Sharif University of Technology)</i></p>

      <p><b>Information Competing Process for Learning Diversified Representations</b><br><i>Jie Hu (Xiamen University) &middot; Rongrong Ji (Xiamen University, China) &middot; ShengChuan Zhang (Xiamen University) &middot; Xiaoshuai Sun (Xiamen University) &middot; Qixiang Ye (University of Chinese Academy of Sciences, China) &middot; Chia-Wen Lin (National Tsing Hua University) &middot; Qi Tian (Huawei Noah’s Ark Lab)</i></p>

      <p><b>GENO -- GENeric Optimization for Classical Machine Learning</b><br><i>Soeren Laue (Friedrich Schiller University Jena / Data Assessment Solutions) &middot; Matthias Mitterreiter (Friedrich Schiller University Jena) &middot; Joachim Giesen (Friedrich-Schiller-Universitat Jena)</i></p>

      <p><b>Conditional Independence Testing using Generative Adversarial Networks</b><br><i>Alexis Bellot (University of Cambridge) &middot; Mihaela van der Schaar (University of Cambridge, Alan Turing Institute and UCLA)</i></p>

      <p><b>Online Stochastic Shortest Path with Bandit Feedback and Unknown Transition Function</b><br><i>Aviv Rosenberg (Tel Aviv University) &middot; Yishay Mansour (Tel Aviv University / Google)</i></p>

      <p><b>Partitioning Structure Learning for Segmented Linear Regression Trees</b><br><i>Xiangyu Zheng (Peking University) &middot; Song Xi Chen (Peking University)</i></p>

      <p><b>A Tensorized Transformer for Language Modeling</b><br><i>Xindian Ma (Tianjin University) &middot; Peng Zhang (Tianjin University) &middot; Shuai Zhang (Tianjin University) &middot; Nan Duan (Microsoft Research) &middot; Yuexian Hou (Tianjin University) &middot; Ming Zhou (Microsoft Research) &middot; Dawei Song (Beijing Institute of Technology)</i></p>

      <p><b>Kernel Stein Tests for Multiple Model Comparison</b><br><i>Jen Ning Lim (Max Planck Institute for Intelligent Systems) &middot; Makoto Yamada (Kyoto University / RIKEN AIP) &middot; Bernhard Schölkopf (MPI for Intelligent Systems) &middot; Wittawat Jitkrittum (Max Planck Institute for Intelligent Systems)</i></p>

      <p><b>Disentangled behavioural representations</b><br><i>Amir Dezfouli (Data61, CSIRO) &middot; Hassan Ashtiani (McMaster University) &middot; Omar Ghattas (CSIRO) &middot; Richard Nock (Data61, the Australian National University and the University of Sydney) &middot; Peter Dayan (Max Planck Institute for Biological Cybernetics) &middot; Cheng Soon Ong (Data61 and ANU)</i></p>

      <p><b>More Is Less: Learning Efficient Video Representations by Temporal Aggregation Module</b><br><i>Quanfu Fan (IBM Research) &middot; Chun-Fu Chen (IBM Research) &middot; Hilde Kuehne (University of Bonn) &middot; Marco Pistoia (IBM Research) &middot; David Cox (MIT-IBM Watson AI Lab)</i></p>

      <p><b>Rethinking the CSC Model for Natural Images</b><br><i>Dror Simon (Technion) &middot; Michael Elad (Technion)</i></p>

      <p><b>Integrating Generative and Discriminative Sparse Kernel Machines for  Multi-class Active Learning</b><br><i>Weishi Shi (Rochester Institute of Technology) &middot; Qi Yu (Rochester Institute of Technology)</i></p>

      <p><b>Learning to Control Self-Assembling Morphologies: A Study of Generalization via Modularity</b><br><i>Deepak Pathak (UC Berkeley) &middot; Christopher Lu (UC Berkeley) &middot; Trevor Darrell (UC Berkeley) &middot; Phillip Isola (Massachusetts Institute of Technology) &middot; Alexei Efros (UC Berkeley)</i></p>

      <p><b>Perceiving the arrow of time in autoregressive motion</b><br><i>Kristof Meding (Max Planck Institute for Intelligent Systems) &middot; Dominik Janzing (Amazon) &middot; Bernhard Schölkopf (MPI for Intelligent Systems) &middot; Felix A. Wichmann (University of Tübingen)</i></p>

      <p><b>DualDICE: Behavior-Agnostic Estimation of Discounted Stationary Distribution Corrections</b><br><i>Ofir Nachum (Google Brain) &middot; Yinlam Chow (DeepMind) &middot; Bo Dai (Google Brain) &middot; Lihong Li (Google Brain)</i></p>

      <p><b>Hyper-Graph-Network Decoders for Block Codes</b><br><i>Eliya Nachmani (Tel Aviv University and Facebook AI Research) &middot; Lior Wolf (Facebook AI Research)</i></p>

      <p><b>Large Scale Markov Decision Processes with Changing Rewards</b><br><i>Adrian Rivera Cardoso (Georgia Tech) &middot; He Wang (Georgia Institute of Technology) &middot; Huan Xu (Georgia Inst. of Technology)</i></p>

      <p><b>Multiview Aggregation for Learning Category-Specific Shape Reconstruction</b><br><i>Srinath Sridhar (Stanford University) &middot; Davis Rempe (Stanford University) &middot; Julien Valentin (Google) &middot; Bouaziz Sofien () &middot; Leonidas J Guibas (stanford.edu)</i></p>

      <p><b>Semi-Parametric Dynamic Contextual Pricing</b><br><i>Virag Shah (Stanford) &middot; Ramesh  Johari (Stanford University) &middot; Jose Blanchet (Stanford University)</i></p>

      <p><b>Nearly Linear-Time, Deterministic Algorithm for Maximizing (Non-Monotone) Submodular Functions Under Cardinality Constraint</b><br><i>Alan Kuhnle (Florida State University)</i></p>

      <p><b>Initialization of ReLUs for Dynamical Isometry</b><br><i>Rebekka Burkholz (Harvard University) &middot; Alina Dubatovka (ETH Zurich)</i></p>

      <p><b>Gradient Information for Representation and Modeling</b><br><i>Jie Ding (University of Minnesota) &middot; Robert Calderbank (Duke University) &middot; Vahid Tarokh (Duke University)</i></p>

      <p><b>SpiderBoost and Momentum: Faster Variance Reduction Algorithms</b><br><i>Zhe Wang (Ohio State University) &middot; Kaiyi Ji (The Ohio State University) &middot; Yi Zhou (University of Utah) &middot; Yingbin Liang (The Ohio State University) &middot; Vahid Tarokh (Duke University)</i></p>

      <p><b>Minimax rates of estimating approximate differential privacy</b><br><i>Xiyang Liu (University of Washington) &middot; Sewoong Oh (University of Washington)</i></p>

      <p><b>Backprop with Approximate Activations for Memory-efficient Network Training</b><br><i>Ayan Chakrabarti (Washington University in St. Louis) &middot; Benjamin Moseley (Carnegie Mellon University)</i></p>

      <p><b>Training Image Estimators without Image Ground Truth</b><br><i>Zhihao Xia (Washington University in St. Louis) &middot; Ayan Chakrabarti (Washington University in St. Louis)</i></p>

      <p><b>Deep Structured Prediction for Facial Landmark Detection</b><br><i>Lisha Chen (Rensselaer Polytechnic Institute) &middot; Hui Su (IBM) &middot; Qiang Ji (Rensselaer Polytechnic Institute)</i></p>

      <p><b>Information-Theoretic Confidence Bounds for Reinforcement Learning</b><br><i>Xiuyuan Lu (Stanford University) &middot; Benjamin Van Roy (Stanford University)</i></p>

      <p><b>Transfer Anomaly Detection by Inferring Latent Domain Representations</b><br><i>Atsutoshi Kumagai (NTT) &middot; Tomoharu Iwata (NTT) &middot; Yasuhiro Fujiwara (NTT Software Innovation Center)</i></p>

      <p><b>Total Least Squares Regression in Input Sparsity Time</b><br><i>Huaian Diao (Northeast Normal University) &middot; Zhao Song (Harvard University & University of Washington) &middot; David Woodruff (Carnegie Mellon University) &middot; Xin Yang (University of Washington)</i></p>

      <p><b>Park: An Open Platform for Learning-Augmented Computer Systems</b><br><i>Hongzi Mao (MIT) &middot; Parimarjan Negi (MIT CSAIL) &middot; Akshay Narayan (MIT CSAIL) &middot; Hanrui Wang (Massachusetts Institute of Technology) &middot; Jiacheng Yang (MIT CSAIL) &middot; Haonan Wang (MIT CSAIL) &middot; Ryan Marcus (MIT CSAIL) &middot; ravichandra addanki (Massachusetts Institute of Technology) &middot; Mehrdad Khani Shirkoohi (MIT) &middot; Songtao He (Massachusetts Institute of Technology) &middot; Vikram Nathan (MIT) &middot; Frank Cangialosi (MIT CSAIL) &middot; Shaileshh Venkatakrishnan (MIT) &middot; Wei-Hung Weng (Massachusetts Institute of Technology) &middot; Song Han (MIT) &middot; Tim Kraska (MIT) &middot; Dr.Mohammad Alizadeh (Massachusetts institute of technology)</i></p>

      <p><b>Adapting Neural Networks for the Estimation of Treatment Effects</b><br><i>Claudia Shi (Columbia University) &middot; David Blei (Columbia University) &middot; Victor Veitch (Columbia University)</i></p>

      <p><b>Learning Transferable Graph Exploration</b><br><i>Hanjun Dai (Georgia Tech) &middot; Yujia Li (DeepMind) &middot; Chenglong Wang (University of Washington) &middot; Rishabh Singh (Google Brain) &middot; Po-Sen Huang (DeepMind) &middot; Pushmeet Kohli (DeepMind)</i></p>

      <p><b>Conformal Prediction Under Covariate Shift</b><br><i>Rina Foygel Barber (University of Chicago) &middot; Emmanuel Candes (Stanford University) &middot; Aaditya Ramdas (CMU) &middot; Ryan Tibshirani (Carnegie Mellon University)</i></p>

      <p><b>Optimal Analysis of Subset-Selection Based L_p Low-Rank Approximation</b><br><i>Chen Dan (Carnegie Mellon University) &middot; Hong Wang (Massachusetts Institute of Technology) &middot; Hongyang Zhang (Carnegie Mellon University) &middot; Yuchen Zhou (University of Wisconsin, Madison) &middot; Pradeep Ravikumar (Carnegie Mellon University)</i></p>

      <p><b>Asymmetric Valleys: Beyond Sharp and Flat Local Minima</b><br><i>Haowei He (Beihang University) &middot; Gao Huang (Tsinghua) &middot; Yang Yuan (Cornell University)</i></p>

      <p><b>Positive-Unlabeled Compression on the Cloud</b><br><i>Yixing Xu (Huawei Noah's Ark Lab) &middot; Yunhe Wang (Noah’s Ark Laboratory, Huawei Technologies Co., Ltd.) &middot; Hanting Chen (Peking University) &middot; Kai Han (Huawei Noah's Ark Lab) &middot; Chunjing XU (Huawei Technologies) &middot; Dacheng Tao (University of Sydney) &middot; Chang Xu (University of Sydney)</i></p>

      <p><b>Direct Estimation of Differential Functional Graphical Model</b><br><i>Boxin Zhao (UChicago) &middot; Sam Wang (UW) &middot; Mladen Kolar (University of Chicago)</i></p>

      <p><b>On the Calibration of Multiclass Classification  with Rejection</b><br><i>Chenri Ni (The University of Tokyo) &middot; Nontawat Charoenphakdee (The University of Tokyo / RIKEN) &middot; Junya Honda (The University of Tokyo / RIKEN) &middot; Masashi Sugiyama (RIKEN / University of Tokyo)</i></p>

      <p><b>Third-Person Visual Imitation Learning via Decoupled Hierarchical Control</b><br><i>Pratyusha Sharma (Carnegie Mellon University) &middot; Deepak Pathak (UC Berkeley) &middot; Abhinav Gupta (Facebook AI Research/CMU)</i></p>

      <p><b>Stagewise Training Accelerates Convergence of Testing Error Over SGD</b><br><i>Zhuoning Yuan (UI-Computer Science) &middot; Yan Yan (the University of Iowa) &middot; Jing Rong (Alibaba) &middot; Tianbao Yang (The University of Iowa)</i></p>

      <p><b>Learning Robust Options by Conditional Value at Risk Optimization</b><br><i>Takuya Hiraoka (NEC) &middot; Takahisa Imagawa (National Institute of Advanced Industrial Science and Technology) &middot; Tatsuya Mori (NEC) &middot; Takashi Onishi (NEC) &middot; Yoshimasa Tsuruoka (The University of Tokyo)</i></p>

      <p><b>Non-asymptotic Analysis of Stochastic Methods for Non-Smooth Non-Convex Regularized Problems</b><br><i>Yi Xu (The University of Iowa) &middot; Jing Rong (Alibaba) &middot; Tianbao Yang (The University of Iowa)</i></p>

      <p><b>On Learning Over-parameterized Neural Networks: A Functional Approximation Prospective</b><br><i>Lili Su (MIT) &middot; Pengkun  Yang (Princeton University)</i></p>

      <p><b>Drill-down: Interactive Retrieval of Complex Scenes using Natural Language Queries</b><br><i>Fuwen Tan (University of Virginia) &middot; Paola Cascante-Bonilla (University of Virginia) &middot; Xiaoxiao Guo (IBM Research) &middot; Hui Wu (IBM Research) &middot; Song Feng (IBM Research) &middot; Vicente Ordonez (University of Virginia)</i></p>

      <p><b>Visual Sequence Learning  in Hierarchical Prediction Networks and Primate Visual Cortex</b><br><i>JIELIN QIU (Shanghai Jiao Tong University) &middot; Ge Huang (Carnegie Mellon University) &middot; Tai Sing Lee (Carnegie Mellon University)</i></p>

      <p><b>Dual Variational Generation for Low Shot Heterogeneous Face Recognition</b><br><i>Chaoyou Fu (Institute of Automation, Chinese Academy of Sciences) &middot; Xiang Wu (Institue of Automation, Chinese Academy of Science) &middot; Yibo Hu (Institute of Automation, Chinese Academy of Sciences) &middot; Huaibo Huang (Institute of Automation, Chinese Academy of Science) &middot; Ran He (NLPR, CASIA)</i></p>

      <p><b>Discovering Neural Wirings</b><br><i>Mitchell N Wortsman (University of Washington, Allen Institute for Artificial Intelligence) &middot; Ali Farhadi (University of Washington, Allen Institute for Artificial Intelligence) &middot; Mohammad Rastegari (Allen Institute for Artificial Intelligence (AI2))</i></p>

      <p><b>On the Optimality of Perturbations in Stochastic and Adversarial Multi-armed Bandit Problems</b><br><i>Baekjin Kim (University of Michigan) &middot; Ambuj Tewari (University of Michigan)</i></p>

      <p><b>Knowledge Extraction with No Observable Data</b><br><i>Jaemin Yoo (Seoul National University) &middot; Minyong Cho (Seoul National University) &middot; Taebum Kim (Seoul National University) &middot; U Kang (Seoul National University)</i></p>

      <p><b>PAC-Bayes under potentially heavy tails</b><br><i>Matthew Holland (Osaka University)</i></p>

      <p><b>One-Shot Object Detection with Co-Attention and Co-Excitation</b><br><i>Ting-I Hsieh (National Tsing Hua University) &middot; Yi-Chen Lo (National Tsing Hua University) &middot; Hwann-Tzong Chen (National Tsing Hua University) &middot; Tyng-Luh Liu (Academia Sinica)</i></p>

      <p><b>Quaternion Knowledge Graph Embeddings</b><br><i>SHUAI ZHANG (University of New South Wales) &middot; Yi Tay (Nanyang Technological University) &middot; Lina Yao (UNSW) &middot; Qi Liu (Facebook AI Research)</i></p>

      <p><b>Glyce: Glyph-vectors for Chinese Character Representations</b><br><i>Yuxian Meng (Shannon.AI) &middot; Wei Wu (Shannon.AI) &middot; Fei Wang (Shannon.AI) &middot; Xiaoya Li (Shannon.AI) &middot; Ping Nie (Shannon.AI) &middot; Fan Yin (Shannon.AI) &middot; Muyu Li (Shannon.AI) &middot; Qinghong  Han (Shannon.AI) &middot; Xiaofei Sun (Shannon.AI) &middot; Jiwei Li (Shannon.AI)</i></p>

      <p><b>Turbo Autoencoder: Deep learning based channel code for point-to-point communication channels</b><br><i>Yihan Jiang (University of Washington Seattle) &middot; Hyeji Kim (Samsung AI Center Cambridge) &middot; Himanshu Asnani (University of Washington, Seattle) &middot; Sreeram Kannan (University of Washington) &middot; Sewoong Oh (University of Washington) &middot; Pramod Viswanath (UIUC)</i></p>

      <p><b>Heterogeneous Graph Learning for Visual Commonsense Reasoning</b><br><i>Weijiang Yu (Sun Yat-sen University) &middot; Jingwen Zhou (Sun Yat-sen University) &middot; Weihao Yu (Sun Yat-sen University) &middot; Xiaodan Liang (Sun Yat-sen University) &middot; Nong Xiao (Sun Yat-sen University)</i></p>

      <p><b>Probabilistic Watershed: Sampling all spanning forests for seeded segmentation and semi-supervised learning</b><br><i>Enrique Fita Sanmartin (Heidelberg University) &middot; Sebastian Damrich (Heidelberg University) &middot; Fred Hamprecht (Heidelberg University)</i></p>

      <p><b>Classification-by-Components: Probabilistic Modeling of Reasoning over a Set of Components</b><br><i>Sascha Saralajew (Dr. Ing. h.c. Porsche AG) &middot; Lars G Holdijk (Radboud University Nijmegen) &middot; Maike Rees (Dr. Ing. h.c. F. Porsche AG) &middot; Ebubekir Asan (Dr. Ing. h.c. F. Porsche AG) &middot; Thomas Villmann (Hochschule Mittweida)</i></p>

      <p><b>Identifying Causal Effects via Context-specific Independence Relations</b><br><i>Santtu Tikka (University of Jyväskylä) &middot; Antti Hyttinen (University of Helsinki) &middot; Juha Karvanen (University of Jyvaskyla)</i></p>

      <p><b>Bridging Machine Learning and Logical Reasoning by Abductive Learning</b><br><i>Wang-Zhou Dai (Imperial College London) &middot; Qiuling Xu (Purdue University) &middot; Yang Yu (Nanjing University) &middot; Zhi-Hua Zhou (Nanjing University)</i></p>

      <p><b>Regret Minimization for Reinforcement Learning by Evaluating the Optimal Bias Function</b><br><i>Zihan Zhang (Tsinghua University) &middot; Xiangyang Ji (Tsinghua University)</i></p>

      <p><b>On the Global Convergence of (Fast) Incremental Expectation Maximization Methods</b><br><i>Belhal Karimi (Ecole Polytechnique) &middot; Hoi-To Wai (Chinese University of Hong Kong) &middot; Eric Moulines (Ecole Polytechnique) &middot; Marc Lavielle (Inria & Ecole Polytechnique)</i></p>

      <p><b>A Linearly Convergent Proximal Gradient Algorithm for Decentralized  Optimization</b><br><i>Sulaiman Alghunaim (UCLA) &middot; Kun Yuan (UCLA) &middot; Ali H. Sayed (Ecole Polytechnique Fédérale de Lausanne)</i></p>

      <p><b>Regularizing Trajectory Optimization with Denoising Autoencoders</b><br><i>Rinu Boney (Aalto University) &middot; Norman Di Palo (Sapienza University of Rome) &middot; Mathias Berglund (Curious AI) &middot; Alexander Ilin (Aalto University) &middot; Juho Kannala (Aalto University) &middot; Antti Rasmus (The Curious AI Company) &middot; Harri Valpola (Curious AI)</i></p>

      <p><b>Learning Hierarchical Priors in VAEs</b><br><i>Alexej Klushyn (Volkswagen Group) &middot; Nutan Chen (Volkswagen Group) &middot; Richard Kurle (Volkswagen Group) &middot; Botond Cseke (Volkswagen Group) &middot; Patrick van der Smagt (Volkswagen Group)</i></p>

      <p><b>Epsilon-Best-Arm Identification in Pay-Per-Reward Multi-Armed Bandits</b><br><i>Sivan Sabato (Ben-Gurion University of the Negev)</i></p>

      <p><b>Safe Exploration for Interactive Machine Learning</b><br><i>Matteo Turchetta (ETH Zurich) &middot; Felix Berkenkamp (ETH Zurich) &middot; Andreas Krause (ETH Zurich)</i></p>

      <p><b>Addressing Failure Detection by Learning Model Confidence</b><br><i>Charles Corbiere (Valeo.ai) &middot; Nicolas THOME (Cnam) &middot; Avner Bar-Hen (CNAM, Paris) &middot; Matthieu Cord (Sorbonne University) &middot; Patrick Pérez (Valeo.ai)</i></p>

      <p><b>Combinatorial Bayesian Optimization using the Graph Cartesian Product</b><br><i>Changyong Oh (University of Amsterdam) &middot; Jakub Tomczak (Qualcomm AI Research) &middot; Efstratios Gavves (University of Amsterdam) &middot; Max Welling (University of Amsterdam / Qualcomm AI Research)</i></p>

      <p><b>Fooling Neural Network Interpretations via Adversarial Model Manipulation</b><br><i>Juyeon Heo (Sungkyunkwan University) &middot; Sunghwan Joo (Sungkyunkwan University) &middot; Taesup Moon (Sungkyunkwan University (SKKU))</i></p>

      <p><b>On Lazy Training in Differentiable Programming</b><br><i>Lénaïc Chizat (INRIA) &middot; Edouard Oyallon (CentraleSupelec) &middot; Francis Bach (INRIA - Ecole Normale Superieure)</i></p>

      <p><b>Quality Aware Generative Adversarial Networks</b><br><i>Parimala Kancharla (Indian Institute of Technology, Hyderabad) &middot; Sumohana S Channappayya (Indian Institute of Technology Hyderabad)</i></p>

      <p><b>Copula-like Variational Inference</b><br><i>Marcel Hirt (University College London) &middot; Petros Dellaportas (University College London, Athens University of Economics and Alan Turing Institute) &middot; Alain Durmus (ENS)</i></p>

      <p><b>Implicit Regularization for Optimal Sparse Recovery</b><br><i>Tomas Vaskevicius (University of Oxford) &middot; Varun Kanade (University of Oxford) &middot; Patrick Rebeschini (University of Oxford)</i></p>

      <p><b>Locally Private Gaussian Estimation</b><br><i>Matthew Joseph (University of Pennsylvania) &middot; Janardhan Kulkarni (Microsoft Research) &middot; Jieming Mao (Google Research) &middot; Steven Wu (Microsoft Research)</i></p>

      <p><b>Multi-mapping Image-to-Image Translation via Learning Disentanglement</b><br><i>Xiaoming Yu (Peking University, Shenzhen Graduate School and  Peng Cheng Laboratory) &middot; Yuanqi Chen (SECE, Peking University) &middot; Shan Liu (Tencent) &middot; Thomas Li (Shenzhen Graduate School, Peking University) &middot; Ge Li (SECE, Shenzhen Graduate School, Peking University)</i></p>

      <p><b>Spatially Aggregated Gaussian Processes with Multivariate Areal Outputs</b><br><i>Yusuke Tanaka (NTT) &middot; Toshiyuki Tanaka (Kyoto University) &middot; Tomoharu Iwata (NTT) &middot; Takeshi Kurashima (NTT Corporation) &middot; Maya Okawa (NTT) &middot; Yasunori Akagi (NTT Service Evolution Laboratories, NTT Corporation) &middot; Hiroyuki Toda (NTT Service Evolution Laboratories, NTT Corporation, Japan)</i></p>

      <p><b>Structured Decoding for Non-Autoregressive Machine Translation</b><br><i>Zhiqing SUN (Peking University) &middot; Zhuohan Li (UC Berkeley) &middot; Haoqing Wang (Peking University) &middot; Di He (Peking University) &middot; Zi Lin (Peking University) &middot; Zhihong Deng (Peking University)</i></p>

      <p><b>Learning Temporal Pose Estimation from Sparsely-Labeled Videos</b><br><i>Gedas Bertasius (Facebook Research) &middot; Christoph Feichtenhofer (Facebook AI Research) &middot; Du Tran (Facebook) &middot; Jianbo Shi (University of Pennsylvania) &middot; Lorenzo Torresani (Facebook AI Research)</i></p>

      <p><b>Greedy InfoMax for Biologically Plausible Self-Supervised Representation Learning</b><br><i>Sindy Löwe (University of Amsterdam) &middot; Peter O'Connor (University of Amsterdam) &middot; Bastiaan Veeling (AMLab - University of Amsterdam)</i></p>

      <p><b>Scalable Gromov-Wasserstein Learning for Graph Partitioning and Matching</b><br><i>Hongteng Xu (Duke University) &middot; Dixin Luo (Duke University) &middot; Lawrence Carin (Duke University)</i></p>

      <p><b>Meta-Reinforced Synthetic Data for One-Shot Fine-Grained Visual Recognition</b><br><i>Satoshi Tsutsui (Indiana University) &middot; Yanwei Fu (Fudan University, Shanghai;  AItrics Inc.  Seoul) &middot; David Crandall (Indiana University)</i></p>

      <p><b>Real-Time Reinforcement Learning</b><br><i>Simon Ramstedt (University of Montreal) &middot; Chris Pal (Montreal Institute for Learning Algorithms, École Polytechnique, Université de Montréal)</i></p>

      <p><b>Robust Multi-agent Counterfactual Prediction</b><br><i>Alexander Peysakhovich (Facebook) &middot; Christian Kroer (Columbia University) &middot; Adam Lerer (Facebook AI Research)</i></p>

      <p><b>Approximate Inference Turns Deep Networks into Gaussian Processes</b><br><i>Mohammad Emtiyaz Khan (RIKEN) &middot; Alexander Immer (EPFL) &middot; Ehsan Abedi (EPFL) &middot; Maciej Jan Korzepa (Technical University of Denmark)</i></p>

      <p><b>Deep Signatures</b><br><i>Patrick Kidger (University of Oxford) &middot; Patric Bonnier (University of Oxford) &middot; Imanol Perez Arribas (University of Oxford) &middot; Cristopher Salvi (University of Oxford) &middot; Terry Lyons (University of Oxford)</i></p>

      <p><b>Individual Regret in Cooperative Nonstochastic Multi-Armed Bandits</b><br><i>Yogev Bar-On (Tel-Aviv University) &middot; Yishay Mansour (Tel Aviv University / Google)</i></p>

      <p><b>Convergent Policy Optimization for Safe Reinforcement Learning</b><br><i>Ming Yu (The University of Chicago, Booth School of Business) &middot; Zhuoran Yang (Princeton University) &middot; Mladen Kolar (University of Chicago) &middot; Zhaoran Wang (Northwestern University)</i></p>

      <p><b>Augmented Neural ODEs</b><br><i>Emilien Dupont (Oxford University) &middot; Arnaud Doucet (Oxford) &middot; Yee Whye Teh (University of Oxford, DeepMind)</i></p>

      <p><b>Thompson Sampling for Multinomial Logit Contextual Bandits</b><br><i>Min-hwan Oh (Columbia University) &middot; Garud Iyengar (Columbia)</i></p>

      <p><b>Backpropagation-Friendly Eigendecomposition</b><br><i>Wei Wang (EPFL) &middot; Zheng Dang (Xi'an Jiaotong University) &middot; Yinlin Hu (EPFL) &middot; Pascal Fua (EPFL, Switzerland) &middot; Mathieu Salzmann (EPFL)</i></p>

      <p><b>FastSpeech: Fast, Robust and Controllable Text to Speech</b><br><i>Yi Ren (Zhejiang University) &middot; Yangjun Ruan (Zhejiang University) &middot; Xu Tan (Microsoft Research) &middot; Tao Qin (Microsoft Research) &middot; Sheng Zhao (Microsoft) &middot; Zhou Zhao (Zhejiang University) &middot; Tie-Yan Liu (Microsoft Research)</i></p>

      <p><b>Ultrametric Fitting by Gradient Descent</b><br><i>Giovanni Chierchia (ESIEE Paris) &middot; Benjamin Perret (ESIEE/PARIS)</i></p>

      <p><b>Distinguishing Distributions When Samples Are Strategically Transformed</b><br><i>Hanrui Zhang (Duke University) &middot; Yu Cheng (Duke University) &middot; Vincent Conitzer (Duke University)</i></p>

      <p><b>Implicit Regularization of Discrete Gradient Dynamics in Deep Linear Neural Networks</b><br><i>Gauthier Gidel (Mila) &middot; Francis Bach (INRIA - Ecole Normale Superieure) &middot; Simon Lacoste-Julien (Mila, Université de Montréal)</i></p>

      <p><b>Deep Set Prediction Networks</b><br><i>Yan Zhang (University of Southampton) &middot; Jonathon Hare (University of Southampton) &middot; Adam Prugel-Bennett (apb@ecs.soton.ac.uk)</i></p>

      <p><b>DppNet: Approximating Determinantal Point Processes with Deep Networks</b><br><i>Zelda Mariet (MIT) &middot; Yaniv Ovadia (Google Inc) &middot; Jasper Snoek (Google Brain)</i></p>

      <p><b>Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control</b><br><i>Sai Zhang (Harvard University) &middot; Qi  Zhang (Amazon) &middot; Jieyu Lin (University of Toronto)</i></p>

      <p><b>Neural Lyapunov Control</b><br><i>Ya-Chien Chang (University of California, San Diego) &middot; Nima Roohi (University of California San Diego) &middot; Sicun Gao (University of California, San Diego)</i></p>

      <p><b>Fully Dynamic Consistent Facility Location</b><br><i>Vincent Cohen-Addad (CNRS & Sorbonne Université) &middot; Niklas Oskar D Hjuler (University of Copenhagen) &middot; Nikos Parotsidis (University of Rome Tor Vergata) &middot; David Saulpic (Ecole normale supérieure) &middot; Chris Schwiegelshohn (Sapienza, University of Rome)</i></p>

      <p><b>A Stickier Benchmark for General-Purpose Language Understanding Systems</b><br><i>Alex Wang (New York University) &middot; Yada Pruksachatkun (New York University) &middot; Nikita Nangia (NYU) &middot; Amanpreet Singh (Facebook) &middot; Julian Michael (University of Washington) &middot; Felix Hill (Google Deepmind) &middot; Omer Levy (Facebook) &middot; Samuel Bowman (New York University)</i></p>

      <p><b>A Flexible Generative Framework for Graph-based Semi-supervised Learning</b><br><i>Jiaqi Ma (University of Michigan) &middot; Weijing Tang (University of Michigan) &middot; Ji Zhu (University of Michigan) &middot; Qiaozhu Mei (University of Michigan)</i></p>

      <p><b>Self-normalization in Stochastic Neural Networks</b><br><i>Georgios Detorakis (University of California, Irvine) &middot; Sourav Dutta (Univ. Notre Dame) &middot;  Abhishek Khanna (Univ. Notre Dame) &middot; Matthew Jerry (Univ. Notre Dame) &middot; Suman Datta (Univ. Notre Dame) &middot; Emre Neftci (Institute for Neural Computation, UCSD)</i></p>

      <p><b>Optimal Decision Tree with Noisy Outcomes</b><br><i>Su Jia (CMU) &middot; viswanath  nagarajan (Univ Michigan, Ann Arbor) &middot; Fatemeh Navidi (University of Michigan) &middot; R Ravi (CMU)</i></p>

      <p><b>Meta-Curvature</b><br><i>Eunbyung Park (UNC Chapel Hill) &middot; Junier Oliva (UNC-Chapel Hill)</i></p>

      <p><b>Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning</b><br><i>Nathan Kallus (Cornell University) &middot; Masatoshi Uehara (Harvard University)</i></p>

      <p><b>KerGM: Kernelized Graph Matching</b><br><i>Zhen Zhang (WASHINGTON UNIVERSITY IN ST.LOUIS) &middot; Yijian Xiang (Washington University in St. Louis) &middot; Lingfei Wu (IBM Research AI) &middot; Bing Xue (Washington University in St. Louis) &middot; Arye Nehorai (WASHINGTON UNIVERSITY IN ST.LOUIS)</i></p>

      <p><b>Transfusion: Understanding Transfer Learning for Medical Imaging</b><br><i>Maithra Raghu (Cornell University and Google Brain) &middot; Chiyuan Zhang (Google Brain) &middot; Jon Kleinberg (Cornell University) &middot; Samy Bengio (Google Research, Brain Team)</i></p>

      <p><b>Adversarial training for free!</b><br><i>Ali Shafahi (University of Maryland) &middot; Mahyar Najibi (University of Maryland) &middot; Mohammad Amin Ghiasi (University of Maryland) &middot; Zheng Xu (Google AI) &middot; John P Dickerson (University of Maryland) &middot; Christoph Studer (Cornell University) &middot; Larry Davis (University of Maryland) &middot; Gavin Taylor (US Naval Academy) &middot; Tom Goldstein (University of Maryland)</i></p>

      <p><b>Communication-Efficient Distributed Learning via Lazily Aggregated Quantized Gradients</b><br><i>Jun Sun (Zhejiang University) &middot; Tianyi Chen (University of Minnesota) &middot; Georgios Giannakis (University of Minnesota) &middot; Zaiyue  Yang (Southern University of Science and Technology)</i></p>

      <p><b>Implicitly learning to reason in first-order logic</b><br><i>Vaishak Belle (University of Edinburgh) &middot; Brendan Juba (Washington University in St. Louis)</i></p>

      <p><b>Kernel-Based Approaches for Sequence Modeling: Connections to Neural Methods</b><br><i>Kevin Liang (Duke University) &middot; Guoyin Wang (Duke University) &middot; Yitong Li (Duke University) &middot; Ricardo Henao (Duke University) &middot; Lawrence Carin (Duke University)</i></p>

      <p><b>PC-Fairness: A Unified Framework for Measuring Causality-based Fairness</b><br><i>Yongkai Wu (University of Arkansas) &middot; Lu Zhang (University of Arkanasa) &middot; Xintao Wu (University of Arkansas) &middot; Hanghang Tong (Arizona State University)</i></p>

      <p><b>Arbicon-Net: Arbitrary Continuous Geometric Transformation Networks for Image Registration</b><br><i>Jianchun Chen (New York University) &middot; Lingjing Wang (New York University) &middot; Xiang Li (New York University) &middot; Yi Fang (New York University)</i></p>

      <p><b>Assessing Disparate Impact of Personalized Interventions: Identifiability and Bounds</b><br><i>Nathan Kallus (Cornell University) &middot; Angela Zhou (Cornell University)</i></p>

      <p><b>The Fairness of Risk Scores Beyond Classification: Bipartite Ranking and the XAUC Metric</b><br><i>Nathan Kallus (Cornell University) &middot; Angela Zhou (Cornell University)</i></p>

      <p><b>HYPE: A Benchmark for Human eYe Perceptual Evaluation of Generative Models</b><br><i>Sharon Zhou (Stanford University) &middot; Mitchell L Gordon (Stanford University) &middot; Ranjay Krishna (Stanford University) &middot; Austin Narcomey (Stanford University) &middot; Li Fei-Fei (Stanford University) &middot; Michael Bernstein (Stanford University)</i></p>

      <p><b>First order expansion of convex regularized estimators</b><br><i>Pierre Bellec (rutgers) &middot; Arun Kuchibhotla (Wharton Statistics)</i></p>

      <p><b>Capacity Bounded Differential Privacy</b><br><i>Kamalika Chaudhuri (UCSD) &middot; Jacob Imola (UCSD) &middot; Ashwin Machanavajjhala (Duke)</i></p>

      <p><b>Universal Boosting Variational Inference</b><br><i>Trevor Campbell (UBC) &middot; Xinglong Li (The University of British Columbia)</i></p>

      <p><b>SGD on Neural Networks Learns Functions of Increasing Complexity</b><br><i>Dimitris Kalimeris (Harvard) &middot; Gal Kaplun (Harvard University) &middot; Preetum Nakkiran (Harvard) &middot; Ben Edelman (Harvard University) &middot; Tristan Yang (Harvard University) &middot; Boaz Barak (Harvard University) &middot; Haofeng Zhang (Harvard University)</i></p>

      <p><b>The Landscape of Non-convex Empirical Risk with Degenerate Population Risk</b><br><i>Shuang Li (Colorado School of Mines) &middot; Gongguo Tang (Colorado School of Mines) &middot; Michael B Wakin (Colorado School of Mines)</i></p>

      <p><b>Making AI Forget You: Data Deletion in Machine Learning</b><br><i>Tony Ginart (Stanford University) &middot; Melody Guan (Stanford University) &middot; Gregory Valiant (Stanford University) &middot; James Zou (Stanford)</i></p>

      <p><b>Practical Differentially Private Top-k Selection with Pay-what-you-get Composition</b><br><i>David Durfee (Georgia Tech) &middot; Ryan Rogers (LinkedIn)</i></p>

      <p><b>Conformalized Quantile Regression</b><br><i>Yaniv Romano (Stanford University) &middot; Evan Patterson (Stanford University) &middot; Emmanuel Candes (Stanford University)</i></p>

      <p><b>Thompson Sampling with Information Relaxation Penalties</b><br><i>Seungki Min (Columbia Business School) &middot; Costis Maglaras (Columbia Business School) &middot; Ciamac C Moallemi (Columbia University)</i></p>

      <p><b>Deep Generalized Method of Moments for Instrumental Variable Analysis</b><br><i>Andrew Bennett (Cornell University) &middot; Nathan Kallus (Cornell University) &middot; Tobias Schnabel (Cornell University)</i></p>

      <p><b>Learning Sample-Specific Models with Low-Rank Personalized Regression</b><br><i>Ben Lengerich (Carnegie Mellon University) &middot; Bryon Aragam (University of Chicago) &middot; Eric Xing (Petuum Inc. /  Carnegie Mellon University)</i></p>

      <p><b>Dance to Music</b><br><i>Hsin-Ying Lee (University of California, Merced) &middot; Xiaodong Yang (NVIDIA Research) &middot; Ming-Yu Liu (Nvidia Research) &middot; Ting-Chun Wang (NVIDIA) &middot; Yu-Ding Lu (UC Merced) &middot; Ming-Hsuan Yang (UC Merced / Google) &middot; Jan Kautz (NVIDIA)</i></p>

      <p><b>Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask</b><br><i>Hattie Zhou (Uber) &middot; Janice Lan (Uber AI Labs) &middot; Rosanne Liu (Uber AI Labs) &middot; Jason Yosinski (Uber AI Labs)</i></p>

      <p><b>Implicit Generation and Modeling with Energy Based Models</b><br><i>Yilun Du (MIT) &middot; Igor Mordatch (OpenAI)</i></p>

      <p><b>Who Learns? Decomposing Learning into Per-Parameter Loss Contribution</b><br><i>Janice Lan (Uber AI Labs) &middot; Rosanne Liu (Uber AI Labs) &middot; Hattie Zhou (Uber) &middot; Jason Yosinski (Uber AI Labs)</i></p>

      <p><b>Predicting the Politics of an Image Using Webly Supervised Data</b><br><i>Christopher Thomas (University of Pittsburgh) &middot; Adriana Kovashka (University of Pittsburgh)</i></p>

      <p><b>Adaptive GNN for Image Analysis and Editing</b><br><i>Lingyu Liang (South China University of Technology) &middot; LianWen Jin (South China University of Technology) &middot; Yong Xu (South China University of Technology)</i></p>

      <p><b>Ultra Fast Medoid Identification via Correlated Sequential Halving</b><br><i>Tavor Z Baharav (Stanford University) &middot; David Tse (Stanford University)</i></p>

      <p><b>Tight Dimension Independent Lower Bound on the Expected Convergence Rate for Diminishing Step Sizes in SGD</b><br><i>PHUONG HA NGUYEN (UCONN) &middot; Lam Nguyen (IBM Thomas J. Watson Research Center) &middot; Marten van Dijk (University of Connecticut)</i></p>

      <p><b>Asymptotics for Sketching in Least Squares Regression</b><br><i>Edgar Dobriban (Stanford University) &middot; Sifan Liu (Tsinghua University)</i></p>

      <p><b>MCP: Learning Composable Hierarchical Control with Multiplicative Compositional Policies</b><br><i>Xue Bin Peng (UC Berkeley) &middot; Michael Chang (University of California, Berkeley) &middot; Grace Zhang (1998) &middot; Pieter Abbeel (UC Berkeley  Covariant) &middot; Sergey Levine (UC Berkeley)</i></p>

      <p><b>Exact inference in structured prediction</b><br><i>Kevin Bello (Purdue University) &middot; Jean Honorio (Purdue University)</i></p>

      <p><b>Coda: An End-to-End Neural Program Decompiler</b><br><i>Cheng Fu (University of California, San Diego) &middot; Huili Chen (UCSD) &middot; Haolan Liu (UCSD) &middot; Xinyun Chen (UC Berkeley) &middot; Yuandong Tian (Facebook AI Research) &middot; Farinaz Koushanfar (UCSD) &middot; Jishen Zhao (UCSD)</i></p>

      <p><b>Bat-G net: Bat-inspired High-Resolution 3D Image Reconstruction using Ultrasonic Echoes</b><br><i>Gunpil Hwang (KAIST) &middot; Seohyeon Kim (KAIST) &middot; Hyeon-Min Bae (KAIST)</i></p>

      <p><b>Painless Stochastic Gradient: Interpolation, Line-Search, and Convergence Rates</b><br><i>Sharan Vaswani (Mila, Université de Montréal) &middot; Aaron Mishkin (University of British Columbia) &middot; Issam Laradji (University of British Columbia) &middot; Mark Schmidt (University of British Columbia) &middot; Gauthier Gidel (Mila) &middot; Simon Lacoste-Julien (Mila, Université de Montréal)</i></p>

      <p><b>Scalable Structure Learning of Continuous-Time Bayesian Networks from Incomplete Data</b><br><i>Dominik Linzner (TU Darmstadt) &middot; Michael Schmidt (TU Darmstadt) &middot; Heinz Koeppl (Technische Universität Darmstadt)</i></p>

      <p><b>Privacy-Preserving Classification of Personal Text Messages with Secure Multi-Party Computation</b><br><i>Devin Reich (University of Washington Tacoma) &middot; Ariel Todoki (University of Washington Tacoma) &middot; Rafael Dowsley (Bar-Ilan University) &middot; Martine De Cock (University of Washington Tacoma) &middot; anderson nascimento (UW)</i></p>

      <p><b>Efficiently Estimating Erdos-Renyi Graphs with Node Differential Privacy</b><br><i>Jonathan Ullman (Northeastern University) &middot; Adam Sealfon (Massachusetts Institute of Technology)</i></p>

      <p><b>Learning Representations for Time Series Clustering</b><br><i>Qianli Ma (South China University of Technology) &middot; Zheng jiawei (South China University of Technology) &middot; Sen Li (South China University of Technology) &middot; Gary W Cottrell (UCSD)</i></p>

      <p><b>Variance Reduced Uncertainty Calibration</b><br><i>Ananya Kumar (Stanford University) &middot; Percy Liang (Stanford University) &middot; Tengyu Ma (Stanford)</i></p>

      <p><b>A Normative Theory for Causal Inference and Bayes Factor Computation in Neural Circuits</b><br><i>Wenhao Zhang (Carnegie Mellon & U. of Pittsburgh) &middot; Si Wu (Peking University) &middot; Brent Doiron (University of Pittsburgh) &middot; Tai Sing Lee (Carnegie Mellon University)</i></p>

      <p><b>Unsupervised Keypoint Learning for Guiding Class-conditional Video Prediction</b><br><i>Yunji Kim (Yonsei University) &middot; Seonghyeon Nam (Yonsei University) &middot; In Cho (Yonsei University) &middot; Seon Joo Kim (Yonsei University)</i></p>

      <p><b>Subspace Attack: Exploiting Promising Subspaces for Query-Efficient Black-box Attacks</b><br><i>Yiwen Guo (Intel Labs China) &middot; Ziang Yan (Tsinghua University) &middot; Changshui Zhang (Tsinghua University)</i></p>

      <p><b>Stochastic Gradient Hamiltonian Monte Carlo Methods with Recursive Variance Reduction</b><br><i>Difan Zou (University of California, Los Angeles) &middot; Pan Xu (University of California, Los Angeles) &middot; Quanquan Gu (UCLA)</i></p>

      <p><b>Learning Latent Process from High-Dimensional Event Sequences via Efficient Sampling</b><br><i>Qitian Wu (Shanghai Jiao Tong University) &middot; Zixuan Zhang (Shanghai Jiao Tong University) &middot; Xiaofeng Gao (Shanghai Jiaotong University) &middot; Junchi Yan (Shanghai Jiao Tong University) &middot; Guihai Chen (Shanghai Jiao Tong University)</i></p>

      <p><b>Cross-sectional Learning of Extremal Dependence among Financial Assets</b><br><i>Xing Yan (The Chinese University of Hong Kong) &middot; Qi Wu (City University of Hong Kong) &middot; Wen Zhang (JD Finance)</i></p>

      <p><b>Principal Component Projection and Regression in Nearly Linear Time through Asymmetric SVRG</b><br><i>Yujia Jin (Stanford University) &middot; Aaron Sidford (Stanford)</i></p>

      <p><b>Compression with Flows via Local Bits-Back Coding</b><br><i>Jonathan Ho (UC Berkeley) &middot; Evan Lohn (University of California, Berkeley) &middot; Pieter Abbeel (UC Berkeley  Covariant)</i></p>

      <p><b>Exact Rate-Distortion in Autoencoders via Echo Noise</b><br><i>Rob Brekelmans (University of Southern Caifornia) &middot; Daniel Moyer (University of Southern California) &middot; Aram Galstyan (USC Information Sciences Inst) &middot; Greg Ver Steeg (University of Southern California)</i></p>

      <p><b>iSplit LBI: Individualized Partial Ranking with Ties via Split LBI</b><br><i>Qianqian Xu (Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences) &middot; Xinwei Sun (MSRA) &middot; Zhiyong Yang (SKLOIS, Institute of Information Engineering, Chinese Academy of Sciences; SCS, University of Chinese Academy of Sciences) &middot; Xiaochun Cao (Chinese Academy of Sciences) &middot; Qingming Huang (University of Chinese Academy of Sciences) &middot; Yuan Yao (Hong Kong Univ. of Science & Technology)</i></p>

      <p><b>Self-Supervised Active Triangulation for 3D Human Pose Reconstruction</b><br><i>Aleksis Pirinen (Lund University) &middot; Erik Gärtner (Lund University) &middot; Cristian Sminchisescu (LTH)</i></p>

      <p><b>MetaQuant: Learning to Quantize by Learning to Penetrate Non-differentiable Quantization</b><br><i>Shangyu Chen (Nanyang Technological University, Singapore) &middot; Wenya Wang (Nanyang Technological University) &middot; Sinno Jialin Pan (Nanyang Technological University, Singapore)</i></p>

      <p><b>Improved Precision and Recall Metric for Assessing Generative Models</b><br><i>Tuomas Kynkäänniemi (NVIDIA; Aalto University) &middot; Tero Karras (NVIDIA) &middot; Samuli Laine (NVIDIA) &middot; Jaakko Lehtinen (Aalto University & NVIDIA) &middot; Timo Aila (NVIDIA Research)</i></p>

      <p><b>A First-order Algorithmic Framework for Distributionally Robust Logistic Regression</b><br><i>Jiajin Li (The Chinese University of Hong Kong) &middot; Sen  Huang (The Chinese University of Hong Kong) &middot; Anthony Man-Cho So (CUHK)</i></p>

      <p><b>PasteGAN: A Semi-Parametric Method to Generate Image from Scene Graph</b><br><i>Yikang LI (The Chinese University of Hong Kong) &middot; Tao Ma (Northwestern Polytechnical University) &middot; Yeqi Bai (Nanyang Technological University) &middot; Nan Duan (Microsoft Research) &middot; Sining Wei (Microsoft Research) &middot; Xiaogang Wang (The Chinese University of Hong Kong)</i></p>

      <p><b>Concomitant Lasso with Repetitions (CLaR): beyond averaging multiple realizations of heteroscedastic noise</b><br><i>Quentin Bertrand (INRIA) &middot; Mathurin Massias (Inria) &middot; Alexandre Gramfort (INRIA, Université Paris-Saclay) &middot; Joseph Salmon (Université de Montpellier)</i></p>

      <p><b>Joint Optimization of Tree-based Index and Deep Model for Recommender Systems</b><br><i>Han Zhu (Alibaba Group) &middot; Daqing Chang (Alibaba Group) &middot; Ziru Xu (Alibaba Group) &middot; Pengye Zhang (Alibaba Group) &middot; Xiang Li (Alibaba Group) &middot; Jie He (Alibaba Group) &middot; Han Li (Alibaba Group) &middot; Jian Xu (Alibaba Group) &middot; Kun Gai (Alibaba Group)</i></p>

      <p><b>Learning Generalizable Device Placement Algorithms for Distributed Machine Learning</b><br><i>ravichandra addanki (Massachusetts Institute of Technology) &middot; Shaileshh Bojja Venkatakrishnan (Massachusetts Institute of Technology) &middot; Shreyan Gupta (MIT) &middot; Hongzi Mao (MIT) &middot; Mohammad Alizadeh (Massachusetts Institute of Technology)</i></p>

      <p><b>Uncoupled Regression from Pairwise Comparison Data</b><br><i>Liyuan Xu (The University of Tokyo / RIKEN) &middot; Junya Honda () &middot; Gang Niu (RIKEN) &middot; Masashi Sugiyama (RIKEN / University of Tokyo)</i></p>

      <p><b>Cross Attention Network for Few-shot Classification</b><br><i>Ruibing Hou (Institute of Computing Technology，Chinese Academy) &middot; Hong Chang (Institute of Computing Technology, Chinese Academy of Sciences) &middot; Bingpeng MA (University of Chinese Academy of Sciences) &middot; Shiguang Shan (Chinese Academy of Sciences) &middot; Xilin Chen (Institute of Computing Technology, Chinese Academy of Sciences)</i></p>

      <p><b>A Nonconvex Approach for Exact and Efficient Multichannel Sparse Blind Deconvolution</b><br><i>Qing Qu (New York University) &middot; Xiao Li (The Chinese University of Hong Kong) &middot; Zhihui Zhu (Johns Hopkins University)</i></p>

      <p><b>SCAN: A Scalable Neural Networks Framework Towards Compact and Efficient Models</b><br><i>Linfeng Zhang (Tsinghua University ) &middot; Zhanhong Tan (Tsinghua University) &middot; Jiebo Song (Institute for Interdisciplinary Information Core Technology) &middot; Jingwei Chen (Tsinghua University) &middot; Chenglong Bao (Tsinghua university) &middot; Kaisheng Ma (Tsinghua University)</i></p>

      <p><b>Revisiting the Bethe-Hessian: Improved Community Detection in Sparse Heterogeneous Graphs</b><br><i>Lorenzo Dall'Amico (GIPSA lab) &middot; Romain Couillet (CentralSupélec) &middot; Nicolas Tremblay (CNRS)</i></p>

      <p><b>Teaching Multiple Concepts to a Forgetful Learner</b><br><i>Anette Hunziker (ETH Zurich and University of Zurich) &middot; Yuxin Chen (Caltech) &middot; Oisin Mac Aodha (California Institute of Technology) &middot; Manuel Gomez Rodriguez (Max Planck Institute for Software Systems) &middot; Andreas Krause (ETH Zurich) &middot; Pietro Perona (California Institute of Technology) &middot; Yisong Yue (Caltech) &middot; Adish Singla (MPI-SWS)</i></p>

      <p><b>Regularized Weighted Low Rank Approximation</b><br><i>Frank Ban (UC Berkeley) &middot; David Woodruff (Carnegie Mellon University) &middot; Richard Zhang (UC Berkeley)</i></p>

      <p><b>Practical and Consistent Estimation of f-Divergences</b><br><i>Paul Rubenstein (MPI for IS) &middot; Olivier Bousquet (Google Brain (Zurich)) &middot; Josip Djolonga (Google Research, Brain Team) &middot; Carlos Riquelme (Google Brain) &middot; Ilya Tolstikhin (MPI for Intelligent Systems)</i></p>

      <p><b>Approximation Ratios of Graph Neural Networks for Combinatorial Problems</b><br><i>Ryoma Sato (Kyoto University) &middot; Makoto Yamada (Kyoto University) &middot; Hisashi Kashima (Kyoto University/RIKEN Center for AIP)</i></p>

      <p><b>Thinning for Accelerating the Learning of Point Processes</b><br><i>Tianbo Li (Nanyang Technological University) &middot; Yiping Ke (Nanyang Technological University)</i></p>

      <p><b>A Prior of a Googol Gaussians: a Tensor Ring Induced Prior for Generative Models</b><br><i>Maxim Kuznetsov (Insilico Medicine) &middot; Daniil Polykovskiy (Insilico Medicine) &middot; Dmitry Vetrov (Higher School of Economics, Samsung AI Center, Moscow) &middot; Alexander Zhebrak (Insilico Medicine)</i></p>

      <p><b>Differentially Private Markov Chain Monte Carlo</b><br><i>Mikko Heikkilä (University of Helsinki) &middot; Joonas Jälkö (Aalto University) &middot; Onur Dikmen (Halmstad University) &middot; Antti Honkela (University of Helsinki)</i></p>

      <p><b>Full-Gradient Representation for Neural Network Visualization</b><br><i>Suraj Srinivas (Idiap Research Institute & EPFL) &middot; François Fleuret (Idiap Research Institute)</i></p>

      <p><b>q-means: A quantum algorithm for unsupervised machine learning</b><br><i>Iordanis Kerenidis (Université Paris Diderot) &middot; Jonas Landman (Université Paris Diderot) &middot; Alessandro Luongo (IRIF - Atos quantum lab) &middot; Anupam Prakash (Université Paris Diderot)</i></p>

      <p><b>Learner-aware Teaching: Inverse Reinforcement Learning with Preferences and Constraints</b><br><i>Sebastian Tschiatschek (Microsoft Research) &middot; Ahana Ghosh (MPI-SWS) &middot; Luis Haug (ETH Zurich) &middot; Rati Devidze (MPI-SWS) &middot; Adish Singla (MPI-SWS)</i></p>

      <p><b>Limitations of the empirical Fisher approximation</b><br><i>Frederik Kunstner (EPFL) &middot; Philipp Hennig (University of Tübingen and MPI for Intelligent Systems Tübingen) &middot; Lukas Balles (University of Tuebingen)</i></p>

      <p><b>Flow-based Image-to-Image Translation with Feature Disentanglement</b><br><i>Ruho Kondo (Toyota Central R&D Labs., Inc.) &middot; Keisuke Kawano (Toyota Central R&D Labs., Inc) &middot; Satoshi Koide (Toyota Central R&D Labs.) &middot; Takuro Kutsuna (Toyota Central R&D Labs. Inc.)</i></p>

      <p><b>Learning dynamic semi-algebraic proofs</b><br><i>Alhussein Fawzi (DeepMind) &middot; Mateusz Malinowski (DeepMind) &middot; Hamza Fawzi (University of Cambridge) &middot; Omar Fawzi (ENS Lyon)</i></p>

      <p><b>Shape and Time Distorsion Loss for Training Deep Time Series Forecasting Models</b><br><i>Vincent LE GUEN (Conservatoire National des Arts et Métiers) &middot; Nicolas THOME (Cnam)</i></p>

      <p><b>Understanding attention in graph neural networks</b><br><i>Boris Knyazev (University of Guelph) &middot; Graham W Taylor (University of Guelph) &middot; Mohamed R. Amer (Robust.AI)</i></p>

      <p><b>Data Cleansing for Models Trained with SGD</b><br><i>Satoshi Hara (Osaka University) &middot; Atsushi Nitanda (The University of Tokyo / RIKEN) &middot; Takanori Maehara (RIKEN AIP)</i></p>

      <p><b>Curvilinear Distance Metric Learning</b><br><i>Shuo Chen (Nanjing University of Science and Technology) &middot; Lei Luo (Pitt) &middot; Jian Yang (Nanjing University of Science and Technology) &middot; Chen Gong (Nanjing University of Science and Technology) &middot; Jun Li (MIT) &middot; Heng Huang (University of Pittsburgh)</i></p>

      <p><b>Semantically-Regularized Logic Graph Embeddings</b><br><i>Xie Yaqi (National University of Singapore) &middot; Ziwei Xu (National University of Singapore) &middot; Kuldeep S Meel (National University of Singapore) &middot; Mohan Kankanhalli (National University of Singapore,) &middot; Harold Soh (National University of Singapore)</i></p>

      <p><b>Modeling Uncertainty by Learning A Hierarchy of Deep Neural Connections</b><br><i>Raanan Y. Rohekar (Intel AI Lab) &middot; Yaniv Gurwicz (Intel AI Lab) &middot; Shami Nisimov (Intel AI Lab) &middot; Gal Novik (Intel AI Lab)</i></p>

      <p><b>Efficient Graph Generation with Graph Recurrent Attention Networks</b><br><i>Renjie Liao (University of Toronto) &middot; Yujia Li (DeepMind) &middot; Yang Song (Stanford University) &middot; Shenlong Wang (University of Toronto) &middot; Will Hamilton (McGill) &middot; David Duvenaud (University of Toronto) &middot; Raquel Urtasun (Uber ATG) &middot; Richard Zemel (Vector Institute/University of Toronto)</i></p>

      <p><b>Beyond Alternating Updates for Matrix Factorization with Inertial Bregman Proximal Gradient Algorithms</b><br><i>Mahesh Chandra Mukkamala (Saarland University) &middot; Peter Ochs (Saarland University)</i></p>

      <p><b>Learning Deep Bilinear Transformation for Fine-grained Image Representation</b><br><i>Heliang Zheng (University of Science and Technology of China) &middot; Jianlong Fu (Microsoft Research) &middot; Zheng-Jun Zha (University of Science and Technology of China) &middot; Jiebo Luo (U. Rochester)</i></p>

      <p><b>Practical Deep Learning with Bayesian Principles</b><br><i>Kazuki Osawa (Tokyo Institute of Technology) &middot; Siddharth Swaroop (University of Cambridge) &middot; Mohammad Emtiyaz Khan (RIKEN) &middot; Anirudh Jain (Indian Institute of Technology (ISM), Dhanbad) &middot; Runa Eschenhagen (University of Osnabrueck) &middot; Richard E Turner (University of Cambridge) &middot; Rio Yokota (Tokyo Institute of Technology, AIST- Tokyo Tech Real World Big-Data Computation Open Innovation Laboratory (RWBC- OIL), National Institute of Advanced Industrial Science and Technology (AIST))</i></p>

      <p><b>Training Language GANs from Scratch</b><br><i>Cyprien de Masson d'Autume (Google DeepMind) &middot; Shakir Mohamed (DeepMind) &middot; Mihaela Rosca (Google DeepMind) &middot; Jack Rae (DeepMind, UCL)</i></p>

      <p><b>Pseudo-Extended Markov chain Monte Carlo</b><br><i>Christopher Nemeth (Lancaster University) &middot; Fredrik Lindsten (Linköping Universituy) &middot; Maurizio Filippone (EURECOM) &middot; James Hensman (PROWLER.io)</i></p>

      <p><b>Differentially Private Bagging: Improved utility and cheaper privacy than subsample-and-aggregate</b><br><i>James Jordon (University of Oxford) &middot; Jinsung Yoon (University of California, Los Angeles) &middot; Mihaela van der Schaar (University of Cambridge, Alan Turing Institute and UCLA)</i></p>

      <p><b>Propagating Uncertainty in Reinforcement Learning via Wasserstein Barycenters</b><br><i>Alberto Maria Metelli (Politecnico di Milano) &middot; Amarildo Likmeta (Politecnico di Milano) &middot; Marcello Restelli (Politecnico di Milano)</i></p>

      <p><b>On Adversarial Mixup Resynthesis</b><br><i>Christopher Beckham (Ecole Polytechnique de Montreal) &middot; Sina Honari (Mila & University of Montreal) &middot; Alex Lamb (UMontreal (MILA)) &middot; vikas verma (Aalto University) &middot; Farnoosh Ghadiri (École Polytechnique de Montréal) &middot; R Devon Hjelm (Microsoft Research) &middot; Yoshua Bengio (Mila) &middot; Chris Pal (MILA, Polytechnique Montréal, Element AI)</i></p>

      <p><b>A Geometric Perspective on Optimal Representations for Reinforcement Learning</b><br><i>Marc Bellemare (Google Brain) &middot; Will Dabney (DeepMind) &middot; Robert Dadashi-Tazehozi (Google Brain) &middot; Adrien Ali Taiga (Google) &middot; Pablo Samuel Castro (Google) &middot; Nicolas Le Roux (Google Brain) &middot; Dale Schuurmans (Google Inc.) &middot; Tor Lattimore (DeepMind) &middot; Clare Lyle (University of Oxford)</i></p>

      <p><b>Learning New Tricks From Old Dogs: Multi-Source Transfer Learning From Pre-Trained Networks</b><br><i>Joshua Lee (Massachusetts Institute of Technology) &middot; Prasanna Sattigeri (IBM Research) &middot; Gregory Wornell (MIT)</i></p>

      <p><b>Understanding and Improving Layer Normalization</b><br><i>Jingjing Xu (Peking University) &middot; Xu Sun (Peking University) &middot; Zhiyuan Zhang (Peking University) &middot; Guangxiang Zhao (Peking University) &middot; Junyang Lin (Alibaba Group)</i></p>

      <p><b>Uncertainty-based Continual Learning with Adaptive Regularization</b><br><i>Hongjoon Ahn (SKKU) &middot; Donggyu Lee (Sungkyunkwan university) &middot; Sungmin Cha (Sungkyunkwan University) &middot; Taesup Moon (Sungkyunkwan University (SKKU))</i></p>

      <p><b>LIIR: Learning Individual Intrinsic Reward in Multi-Agent Reinforcement Learning</b><br><i>Yali Du (University of Technology Sydney) &middot; Lei Han (Rutgers University) &middot; Meng Fang (Tencent) &middot; Ji Liu (University of Rochester, Tencent AI lab) &middot; Tianhong Dai (Imperial College London) &middot; Dacheng Tao (University of Sydney)</i></p>

      <p><b>U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging</b><br><i>Mathias Perslev (University of Copenhagen) &middot; Michael H Jensen (University of Copehagen) &middot; Sune Darkner (University of Copenhagen, Denmark) &middot; Poul Jørgen Jennum (Danish Center for Sleep Medicine, Rigshospitalet) &middot; Christian Igel (University of Copenhagen)</i></p>

      <p><b>Massively scalable Sinkhorn distances via the Nyström method</b><br><i>Jason Altschuler (MIT) &middot; Francis Bach (INRIA - Ecole Normale Superieure) &middot; Alessandro Rudi (INRIA, Ecole Normale Superieure) &middot; Jonathan Weed (MIT)</i></p>

      <p><b>Double Quantization for Communication-Efficient Distributed Optimization</b><br><i>Yue Yu (Tsinghua University) &middot; Jiaxiang Wu (Tencent AI Lab) &middot; Longbo Huang (IIIS, Tsinghua Univeristy)</i></p>

      <p><b>Globally optimal score-based learning of directed acyclic graphs in high-dimensions</b><br><i>Bryon Aragam (University of Chicago) &middot; Arash Amini (UCLA) &middot; Qing Zhou (UCLA)</i></p>

      <p><b>Multi-relational Poincaré Graph Embeddings</b><br><i>Ivana Balazevic (University of Edinburgh) &middot; Carl Allen (University of Edinburgh) &middot; Timothy Hospedales (University of Edinburgh)</i></p>

      <p><b>No-Press Diplomacy: Modeling Multi-Agent Gameplay</b><br><i>Philip Paquette (Université de Montréal - MILA) &middot; Yuchen Lu (University of Montreal) &middot; SETON STEVEN BOCCO (MILA - Université de Montréal) &middot; Max Smith (University of Michigan) &middot; Satya O.-G. (MILA) &middot; Jonathan K. Kummerfeld (University of Michigan) &middot; Joelle Pineau (McGill University) &middot; Satinder Singh (University of Michigan) &middot; Aaron Courville (U. Montreal)</i></p>

      <p><b>State Aggregation Learning from Markov Transition Data</b><br><i>Yaqi Duan (Princeton University) &middot; Tracy Ke (Harvard University) &middot; Mengdi Wang (Princeton University)</i></p>

      <p><b>Disentangling Influence: Using disentangled representations to audit model predictions</b><br><i>Charles Marx (Haverford College) &middot; Richard Phillips (Haverford College) &middot; Sorelle Friedler (Haverford College) &middot; Carlos  Scheidegger (The University of Arizona) &middot; Suresh Venkatasubramanian (University of Utah)</i></p>

      <p><b>Successor Uncertainties: Exploration and Uncertainty in Temporal Difference Learning</b><br><i>David Janz (University of Cambridge) &middot; Jiri Hron (University of Cambridge) &middot; Przemysław Mazur (Wayve) &middot; Katja Hofmann (Microsoft Research) &middot; José Miguel Hernández-Lobato (University of Cambridge) &middot; Sebastian Tschiatschek (Microsoft Research)</i></p>

      <p><b>Partially Encrypted Deep Learning using Functional Encryption</b><br><i>Theo Ryffel (École Normale Supérieure) &middot; David Pointcheval (École Normale Supérieure) &middot; Francis Bach (INRIA - Ecole Normale Superieure) &middot; Edouard Dufour-Sans (Carnegie Mellon University) &middot; Romain Gay (UC Berkeley)</i></p>

      <p><b>Decentralized Cooperative Stochastic Bandits</b><br><i>David Martínez-Rubio (University of Oxford) &middot; Varun Kanade (University of Oxford) &middot; Patrick Rebeschini (University of Oxford)</i></p>

      <p><b>Statistical bounds for entropic optimal transport: sample complexity and the central limit theorem</b><br><i>Gonzalo Mena (Harvard) &middot; Jonathan Weed (MIT)</i></p>

      <p><b>Efficient Deep Approximation of GMMs</b><br><i>Shirin Jalali (Nokia Bell Labs) &middot; Carl Nuzman (Nokia Bell Labs) &middot; Iraj Saniee (Nokia Bell Labs)</i></p>

      <p><b>Learning low-dimensional state embeddings and metastable clusters from time series data</b><br><i>Yifan Sun (Carnegie Mellon University) &middot; Yaqi Duan (Princeton University) &middot; Hao Gong (Princeton University) &middot; Mengdi Wang (Princeton University)</i></p>

      <p><b>Exploiting Local and Global Structure for Point Cloud Semantic Segmentation with Contextual Point Representations</b><br><i>Xu Wang (Shenzhen University) &middot; Jingming He (Shenzhen University) &middot; Lin Ma (Tencent AI Lab)</i></p>

      <p><b>Scalable Bayesian dynamic covariance modeling with variational Wishart and inverse Wishart processes</b><br><i>Creighton Heaukulani (No Affiliation) &middot; Mark van der Wilk (PROWLER.io)</i></p>

      <p><b>Kernel Instrumental Variable Regression</b><br><i>Rahul Singh (MIT) &middot; Maneesh Sahani (Gatsby Unit, UCL) &middot; Arthur Gretton (Gatsby Unit, UCL)</i></p>

      <p><b>Symmetry-Based Disentangled Representation Learning requires Interaction with Environments</b><br><i>Hugo Caselles-Dupré (Flowers Laboaratory (ENSTA ParisTech & INRIA) & Softbank Robotics Europe) &middot; Michael Garcia Ortiz (SoftBank Robotics Europe) &middot; David Filliat (ENSTA)</i></p>

      <p><b>Fast Efficient Hyperparameter Tuning for Policy Gradient Methods</b><br><i>Supratik Paul (University of Oxford) &middot; Vitaly Kurin (RWTH Aachen University) &middot; Shimon Whiteson (University of Oxford)</i></p>

      <p><b>Offline Contextual Bayesian Optimization</b><br><i>Ian Char (Carnegie Mellon University) &middot; Youngseog Chung (Carnegie Mellon University) &middot; Willie Neiswanger (Carnegie Mellon University) &middot; Kirthevasan Kandasamy (Carnegie Mellon University) &middot; Oak Nelson (Princeton Plasma Physics Lab) &middot; Mark Boyer (Princeton Plasma Physics Lab) &middot;  Egemen Kolemen (Princeton Plasma Physics Lab) &middot; Jeff Schneider (Carnegie Mellon University)</i></p>

      <p><b>Making the Cut: A Bandit-based Approach to Tiered Interviewing</b><br><i>Candice Schumann (University of Maryland) &middot; Zhi Lang (University of Maryland, College Park) &middot; Jeffrey Foster (Tufts University) &middot; John P Dickerson (University of Maryland)</i></p>

      <p><b>Unsupervised Scalable Representation Learning for Multivariate Time Series</b><br><i>Jean-Yves Franceschi (Sorbonne Université) &middot; Aymeric Dieuleveut (EPFL) &middot; Martin Jaggi (EPFL)</i></p>

      <p><b>A state-space model for inferring effective connectivity of latent neural dynamics from simultaneous EEG/fMRI</b><br><i>Tao Tu (Columbia University) &middot; John Paisley (Columbia University) &middot; Stefan Haufe (Charité – Universitätsmedizin Berlin) &middot; Paul Sajda (Columbia University)</i></p>

      <p><b>End to end learning and optimization on graphs</b><br><i>Bryan Wilder (University of Southern California) &middot; Eric Ewing (University of Southern California) &middot; Bistra Dilkina (University of Southern California) &middot; Milind Tambe (USC)</i></p>

      <p><b>Game Design for Eliciting Distinguishable Behavior</b><br><i>Fan Yang (Carnegie Mellon University) &middot; Liu Leqi (Carnegie Mellon University) &middot; Yifan Wu (Carnegie Mellon University) &middot; Zachary Lipton (Carnegie Mellon University) &middot; Pradeep Ravikumar (Carnegie Mellon University) &middot; Tom M Mitchell (Carnegie Mellon University) &middot; William Cohen (Google AI)</i></p>

      <p><b>When does label smoothing help?</b><br><i>Rafael Müller (Google Brain) &middot; Simon Kornblith (Google Brain) &middot; Geoffrey E Hinton (Google & University of Toronto)</i></p>

      <p><b>Finite-Time Performance Bounds and Adaptive Learning Rate Selection for Two Time-Scale Reinforcement Learning</b><br><i>Harsh Gupta (University of Illinois at Urbana-Champaign) &middot; R. Srikant (University of Illinois at Urbana-Champaign) &middot; Lei Ying (ASU)</i></p>

      <p><b>Rethinking Deep Neural Network Ownership Verification: Embedding Passports to Defeat Ambiguity Attacks</b><br><i>Lixin Fan (WeBank AI Lab) &middot; Kam Woh Ng (University of Malaya) &middot; Chee Seng Chan (University of Malaya)</i></p>

      <p><b>Scalable Spike Source Localization in Extracellular Recordings using Amortized Variational Inference</b><br><i>Cole Hurwitz (University of Edinburgh) &middot; Kai Xu (University of Ediburgh) &middot; Akash Srivastava (MIT–IBM Watson AI Lab) &middot; Alessio Buccino (University of Oslo) &middot; Matthias Hennig (University of Edinburgh)</i></p>

      <p><b>Optimal Sketching for Kronecker Product Regression and Low Rank Approximation</b><br><i>Huaian Diao (Northeast Normal University) &middot; Rajesh Jayaram (Carnegie Mellon University) &middot; Zhao Song (UT-Austin) &middot; Wen Sun (Microsoft Research) &middot; David Woodruff (Carnegie Mellon University)</i></p>

      <p><b>Distribution-Independent PAC Learning of Halfspaces with Massart Noise</b><br><i>Ilias Diakonikolas (USC) &middot; Themis Gouleakis (MPI) &middot; Christos Tzamos (Microsoft Research)</i></p>

      <p><b>The Convergence Rate of Neural Networks for Learned Functions of Different Frequencies</b><br><i>Basri Ronen (Weizmann Inst.) &middot; David Jacobs (University of Maryland, USA) &middot; Yoni Kasten (Weizmann Institute) &middot; Shira Kritchman (Weizmann Institute)</i></p>

      <p><b>Online Learning for Auxiliary Task Weighting for Reinforcement Learning</b><br><i>Xingyu Lin (Carnegie Mellon University) &middot; Harjatin Baweja (CMU) &middot; George Kantor (CMU) &middot; David Held (CMU)</i></p>

      <p><b>Blocking Bandits</b><br><i>Soumya Basu (University of Texas at Austin) &middot; Rajat Sen (University of Texas at Austin) &middot; Sujay Sanghavi (UT-Austin) &middot; Sanjay Shakkottai (University of Texas at Austin)</i></p>

      <p><b>Global Convergence of Least Squares EM for Demixing Two Log-Concave Densities</b><br><i>Wei Qian (Cornell Univeristy) &middot; Yuqian Zhang (Cornell University) &middot; Yudong Chen (Cornell University)</i></p>

      <p><b>Prior-Free Dynamic Auctions with Low Regret Buyers</b><br><i>Yuan Deng (Duke University) &middot; Jon Schneider (Google Research) &middot; Balasubramanian Sivan (Google Research)</i></p>

      <p><b>On Single Source Robustness in Deep Fusion Models</b><br><i>Taewan Kim (University of Texas at Austin) &middot; Joydeep Ghosh (UT Austin)</i></p>

      <p><b>Policy Evaluation with Latent Confounders via Optimal Balance</b><br><i>Andrew Bennett (Cornell University) &middot; Nathan Kallus (Cornell University)</i></p>

      <p><b>Think Globally, Act Locally: A Deep Neural Network Approach to High-Dimensional Time Series Forecasting</b><br><i>Rajat Sen (University of Texas at Austin) &middot; Hsiang-Fu Yu (Amazon) &middot; Inderjit S Dhillon (UT Austin & Amazon)</i></p>

      <p><b>Adaptive Cross-Modal Few-shot Learning</b><br><i>Chen Xing (Montreal Institute of Learning Algorithms) &middot; Negar Rostamzadeh (Elemenet AI) &middot; Boris Oreshkin (Element AI) &middot; Pedro O. Pinheiro (Element AI)</i></p>

      <p><b>Spectral Modification of Graphs for Improved Spectral Clustering</b><br><i>Ioannis Koutis (New Jersey Institute of Technology) &middot; Huong Le (NJIT)</i></p>

      <p><b>Hyperbolic Graph Convolutional Neural Networks</b><br><i>Zhitao Ying (Stanford University) &middot; Ines Chami (Stanford University) &middot; Christopher Ré (Stanford) &middot; Jure Leskovec (Stanford University and Pinterest)</i></p>

      <p><b>Cost Effective Active Search</b><br><i>Shali Jiang (Washington University in St. Louis) &middot; Roman Garnett (Washington University in St. Louis) &middot; Benjamin Moseley (Carnegie Mellon University)</i></p>

      <p><b>Exploration Bonus for Regret Minimization in Discrete and Continuous Average Reward MDPs</b><br><i>Jian QIAN (INRIA Lille - Sequel Team) &middot; Ronan Fruit (Inria Lille) &middot; Matteo Pirotta (Facebook AI Research) &middot; Alessandro Lazaric (Facebook Artificial Intelligence Research)</i></p>

      <p><b>Hybrid 8-bit Floating Point (HFP8) Training and Inference for Deep Neural Networks</b><br><i>Xiao Sun (IBM) &middot; Jungwook Choi (Hanyang University) &middot; Chia-Yu Chen (IBM research) &middot; Naigang Wang (IBM T. J. Watson Research Center) &middot; Swagath Venkataramani (IBM Research) &middot; Vijayalakshmi (Viji) Srinivasan (IBM TJ Watson) &middot; Xiaodong Cui (IBM T. J. Watson Research Center) &middot; Wei Zhang (IBM T.J.Watson Research Center) &middot; Kailash Gopalakrishnan (IBM Research)</i></p>

      <p><b>A Stratified Approach to Robustness for Randomly Smoothed Classifiers</b><br><i>Guang-He Lee (MIT) &middot; Yang Yuan (MIT) &middot; Shiyu Chang (IBM T.J. Watson Research Center) &middot; Tommi Jaakkola (MIT)</i></p>

      <p><b>Poisson-Minibatching for Gibbs Sampling with Convergence Rate Guarantees</b><br><i>Ruqi Zhang (Cornell University) &middot; Christopher De Sa (Cornell)</i></p>

      <p><b>One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers</b><br><i>Ari Morcos (Facebook AI Research) &middot; Haonan Yu (Facebook AI Research) &middot; Michela Paganini (Facebook) &middot; Yuandong Tian (Facebook AI Research)</i></p>

      <p><b>Breaking the Glass Ceiling for Embedding-Based Classifiers for Large Output Spaces</b><br><i>Chuan Guo (Cornell University) &middot; Ali Mousavi (Google Brain) &middot; Xiang Wu (Google) &middot; Daniel Holtmann-Rice (Google Inc) &middot; Satyen Kale (Google) &middot; Sashank Reddi (Google) &middot; Sanjiv Kumar (Google Research)</i></p>

      <p><b>Fair Algorithms for Clustering</b><br><i>Maryam Negahbani (Dartmouth College) &middot; Deeparnab Chakrabarty (Dartmouth) &middot; Nicolas Flores (Dartmouth College) &middot; Suman Bera (UC Santa Cruz)</i></p>

      <p><b>Learning Mean-Field Games</b><br><i>Xin Guo (University of California, Berkeley) &middot; Anran Hu (University of Californian, Berkeley (UC Berkeley)) &middot; Renyuan Xu (UC Berkeley) &middot; Junzi Zhang (Stanford University)</i></p>

      <p><b>SpArSe: Sparse Architecture Search for CNNs on Resource-Constrained Microcontrollers</b><br><i>Igor Fedorov (Arm Research) &middot; Ryan Adams (Princeton University) &middot; Matthew Mattina (ARM) &middot; Paul Whatmough (Arm Research)</i></p>

      <p><b>Deep imitation learning for molecular inverse problems</b><br><i>Eric Jonas (University of Chicago)</i></p>

      <p><b>Visual Concept-Metaconcept Learning</b><br><i>Chi Han (Tsinghua University) &middot; Jiayuan Mao (MIT) &middot; Chuang Gan (MIT-IBM Watson AI Lab) &middot; Josh Tenenbaum (MIT) &middot; Jiajun Wu (MIT)</i></p>

      <p><b>Adaptive Video-to-Video Synthesis via Network Weight Generation</b><br><i>Ting-Chun Wang (NVIDIA) &middot; Ming-Yu Liu (Nvidia Research) &middot; Andrew Tao (Nvidia Corporation) &middot; Guilin Liu (NVIDIA) &middot; Bryan Catanzaro (NVIDIA) &middot; Jan Kautz (NVIDIA)</i></p>

      <p><b>Neural Similarity Learning</b><br><i>Weiyang Liu (Georgia Institute of Technology) &middot; Zhen Liu (Georgia Institute of Technology) &middot; James M Rehg (Georgia Tech) &middot; Le Song (Ant Financial & Georgia Institute of Technology)</i></p>

      <p><b>Ordered Memory</b><br><i>Yikang Shen (Mila, University of Montreal, MSR Montreal) &middot; Shawn Tan (Mila) &middot; SeyedArian Hosseini (Iran University of Science and Technology) &middot; Zhouhan Lin (MILA) &middot; Alessandro Sordoni (Microsoft Research) &middot; Aaron Courville (U. Montreal)</i></p>

      <p><b>MixMatch: A Holistic Approach to Semi-Supervised Learning</b><br><i>David Berthelot (Google Brain) &middot; Nicholas Carlini (Google) &middot; Ian Goodfellow (Google Brain) &middot; Nicolas Papernot () &middot; Avital Oliver (Google Brain) &middot; Colin A Raffel (Google Brain)</i></p>

      <p><b>Deep Multivariate Quantiles for Novelty Detection</b><br><i>Jingjing Wang (University of Waterloo) &middot; Sun Sun (University of Waterloo) &middot; Yaoliang Yu (University of Waterloo)</i></p>

      <p><b>Fast Parallel Algorithms for Statistical Subset Selection Problems</b><br><i>Sharon Qian (Harvard) &middot; Yaron Singer (Harvard University)</i></p>

      <p><b>PHYRE: A New Benchmark for Physical Reasoning</b><br><i>Anton Bakhtin (Facebook AI Research) &middot; Laurens van der Maaten (Facebook) &middot; Justin Johnson (Facebook AI Research) &middot; Laura  Gustafson (Facebook AI Research) &middot; Ross Girshick (FAIR)</i></p>

      <p><b>How many variables should be entered in a principal component regression equation?</b><br><i>Ji Xu (Columbia University) &middot; Daniel Hsu (Columbia University)</i></p>

      <p><b>Factor Group-Sparse Regularization for Efficient Low-Rank Matrix Recovery</b><br><i>Jicong Fan (Cornell University) &middot; Lijun Ding (Cornell University) &middot; Yudong Chen (Cornell University) &middot; Madeleine Udell (Cornell University)</i></p>

      <p><b>Mutually Regressive Point Processes</b><br><i>Ifigeneia Apostolopoulou (Carnegie Mellon University) &middot; Scott Linderman (Stanford University) &middot; Kyle Miller (Carnegie Mellon University) &middot; Artur Dubrawski (Carnegie Mellon University)</i></p>

      <p><b>Data-driven Estimation of Sinusoid Frequencies</b><br><i>Gautier Izacard (Ecole Polytechnique) &middot; Sreyas Mohan (NYU) &middot; Carlos Fernandez-Granda (NYU)</i></p>

      <p><b>E2-Train: Energy-Efficient Deep Network Training with Data-, Model-, and Algorithm-Level Saving</b><br><i>Ziyu Jiang (Texas A&M University) &middot; Yue Wang (Rice University) &middot; Xiaohan Chen (Texas A&M University) &middot; Pengfei Xu (Rice University) &middot; Yang Zhao (Rice University) &middot; Yingyan Lin (Rice University) &middot; Zhangyang Wang (TAMU)</i></p>

      <p><b>ANODEV2: A Coupled Neural ODE Framework</b><br><i>Tianjun Zhang (University of California, Berkeley) &middot; Zhewei Yao (UC Berkeley) &middot; Amir Gholami (University of California, Berkeley) &middot; Joseph Gonzalez (UC Berkeley) &middot; Kurt Keutzer (EECS, UC Berkeley) &middot; Michael W Mahoney (UC Berkeley) &middot; George Biros (University of Texas at Austin)</i></p>

      <p><b>Estimating Entropy of Distributions in Constant Space</b><br><i>Jayadev Acharya (Cornell University) &middot; Sourbh Bhadane (Cornell University) &middot; Piotr Indyk (MIT) &middot; Ziteng Sun (Cornell University)</i></p>

      <p><b>On the Utility of Learning about Humans for Human-AI Coordination</b><br><i>Micah Carroll (UC Berkeley) &middot; Rohin Shah (UC Berkeley) &middot; Mark Ho (UC Berkeley) &middot; Thomas Griffiths (Princeton University) &middot; Sanjit Seshia (UC Berkeley) &middot; Pieter Abbeel (UC Berkeley  Covariant) &middot; Anca Dragan (UC Berkeley)</i></p>

      <p><b>Efficient Regret Minimization Algorithm for Extensive-Form Correlated Equilibrium</b><br><i>Gabriele Farina (Carnegie Mellon University) &middot; Chun Kai Ling (Carnegie Mellon University) &middot; Fei Fang (Carnegie Mellon University) &middot; Tuomas Sandholm (Carnegie Mellon University)</i></p>

      <p><b>Learning in Generalized  Linear Contextual Bandits with Stochastic Delays</b><br><i>Zhengyuan Zhou (Stanford University) &middot; Renyuan Xu (UC Berkeley) &middot; Jose Blanchet (Stanford University)</i></p>

      <p><b>Empirically Measuring Concentration: Fundamental Limits on Intrinsic Robustness</b><br><i>Saeed Mahloujifar (University of Virginia) &middot; Xiao Zhang (University of Virginia) &middot; Mohammad Mahmoody (University of Virginia) &middot; David Evans (University of Virginia)</i></p>

      <p><b>Optimistic Regret Minimization for Extensive-Form Games via Dilated Distance-Generating Functions</b><br><i>Gabriele Farina (Carnegie Mellon University) &middot; Christian Kroer (Columbia University) &middot; Tuomas Sandholm (Carnegie Mellon University)</i></p>

      <p><b>On Learning Non-Convergent Non-Persistent Short-Run MCMC Toward Energy-Based Model</b><br><i>Erik Nijkamp (UCLA) &middot; Mitch Hill (UCLA Department of Statistics) &middot; Song-Chun Zhu (UCLA) &middot; Ying Nian Wu (University of California, Los Angeles)</i></p>

      <p><b>Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting</b><br><i>Shiyang Li (UCSB) &middot; Xiaoyong Jin (UCSB) &middot; Yao Xuan (UCSB) &middot; Xiyou Zhou (UCSB) &middot; Wenhu Chen (University of California, Santa Barbara) &middot; Yu-Xiang Wang (UC Santa Barbara) &middot; Xifeng Yan (UCSB)</i></p>

      <p><b>On the Accuracy of Influence Functions for Measuring Group Effects</b><br><i>Pang Wei W Koh (Stanford University) &middot; Kai-Siang Ang (Stanford University) &middot; Hubert Teo (Stanford University) &middot; Percy Liang (Stanford University)</i></p>

      <p><b>Face Reconstruction from Voice using Generative Adversarial Networks</b><br><i>Yandong Wen (Carnegie Mellon University) &middot; Bhiksha Raj (Carnegie Mellon University) &middot; Rita Singh (Carnegie Mellon University)</i></p>

      <p><b>Incremental Few-Shot Learning with Attention Attractor Networks</b><br><i>Mengye Ren (University of Toronto / Uber ATG) &middot; Renjie Liao (University of Toronto) &middot; Ethan Fetaya (University of Toronto) &middot; Richard Zemel (Vector Institute/University of Toronto)</i></p>

      <p><b>On Testing for Biases in Peer Review</b><br><i>Ivan Stelmakh (Carnegie Mellon University) &middot; Nihar Shah (CMU) &middot; Aarti Singh (CMU)</i></p>

      <p><b>Learning Disentangled Representation for Robust Person Re-identification</b><br><i>Chanho Eom (Yonsei University) &middot; Bumsub Ham (Yonsei University)</i></p>

      <p><b>Balancing Efficiency and Fairness in On-Demand Ridesourcing</b><br><i>Nixie Lesmana (Nanyang Technological University) &middot; Xuan Zhang (Shanghai Jiaotong University) &middot; Xiaohui Bei (Nanyang Technological University)</i></p>

      <p><b>Latent Ordinary Differential Equations for Irregularly-Sampled Time Series</b><br><i>Yulia Rubanova (University of Toronto) &middot; Tian Qi Chen (U of Toronto) &middot; David Duvenaud (University of Toronto)</i></p>

      <p><b>Deep RGB-D Canonical Correlation Analysis For Sparse Depth Completion</b><br><i>Yiqi Zhong (University of Southern California) &middot; Cho-Ying Wu (Univ. of Southern California) &middot; Suya You (US Army Research Laboratory) &middot; Ulrich Neumann (USC)</i></p>

      <p><b>Input Similarity from the Neural Network Perspective</b><br><i>Guillaume Charpiat (INRIA) &middot; Nicolas Girard (Inria Sophia-Antipolis) &middot; Loris Felardos (INRIA) &middot; Yuliya Tarabalka (Inria Sophia-Antipolis)</i></p>

      <p><b>Adaptive Sequence Submodularity</b><br><i>Marko Mitrovic (Yale University) &middot; Ehsan Kazemi (Yale) &middot; Moran Feldman (Open University of Israel) &middot; Andreas Krause (ETH Zurich) &middot; Amin Karbasi (Yale)</i></p>

      <p><b>Weight Agnostic Neural Networks</b><br><i>Adam Gaier (Bonn-Rhein-Sieg University of Applied Sciences) &middot; David Ha (Google Brain)</i></p>

      <p><b>Learning to Predict Without Looking Ahead: World Models Without Forward Prediction</b><br><i>Daniel Freeman (Google Brain) &middot; David Ha (Google Brain) &middot; Luke Metz (Google Brain)</i></p>

      <p><b>Reducing the variance in online optimization by transporting past gradients</b><br><i>Sébastien Arnold (USC) &middot; Pierre-Antoine Manzagol (Google) &middot; Reza Harikandeh (UBC) &middot; Ioannis Mitliagkas (Mila & University of Montreal) &middot; Nicolas Le Roux (Google Brain)</i></p>

      <p><b>Characterizing Bias in Classifiers using Generative Models</b><br><i>Daniel McDuff (Microsoft Research) &middot; Shuang Ma (SUNY Buffalo) &middot; Yale Song (Microsoft) &middot; Ashish Kapoor (Microsoft Research)</i></p>

      <p><b>Optimal Stochastic and Online Learning with Individual Iterates</b><br><i>Yunwen Lei (Southern University of Science and Technology) &middot; Peng Yang (Southern University of Science and Technology) &middot; Ke Tang (Southern University of Science and Technology) &middot; Ding-Xuan Zhou (City University of Hong Kong)</i></p>

      <p><b>Policy Learning for Fairness in Ranking</b><br><i>Ashudeep Singh (Cornell University) &middot; Thorsten Joachims (Cornell)</i></p>

      <p><b>Off-Policy Evaluation of Generalization for Deep Q-Learning in Binary Reward Tasks</b><br><i>Alexander Irpan (Google Brain) &middot; Kanishka Rao (Google) &middot; Konstantinos Bousmalis (DeepMind) &middot; Chris Harris (Google) &middot; Julian Ibarz (Google Inc.) &middot; Sergey Levine (Google)</i></p>

      <p><b>Regularized Gradient Boosting</b><br><i>Corinna Cortes (Google Research) &middot; Mehryar Mohri (Courant Inst. of Math. Sciences & Google Research) &middot; Dmitry Storcheus (Google Research)</i></p>

      <p><b>Efficient Probabilistic Inference in the Quest for Physics Beyond the Standard Model</b><br><i>Atilim Gunes Baydin (University of Oxford) &middot; Lei Shao (Intel Corporation) &middot; Wahid Bhimji (Berkeley lab) &middot; Lukas Heinrich (New York University) &middot; Saeid Naderiparizi (University of British Columbia) &middot; Andreas Munk (University of British Columbia) &middot; Jialin Liu (Lawrence Berkeley National Lab) &middot; Bradley J Gram-Hansen (University of Oxford) &middot; Gilles Louppe (University of Liège) &middot; Lawrence Meadows (Intel Corporation) &middot; Philip Torr (University of Oxford) &middot; Victor Lee (Intel Corporation) &middot; Kyle Cranmer (New York University) &middot; Mr. Prabhat (LBL/NERSC) &middot; Frank Wood (University of British Columbia)</i></p>

      <p><b> Markov Random Fields for Collaborative Filtering</b><br><i>Harald Steck (Netflix)</i></p>

      <p><b>A Step Toward Quantifying Independently Reproducible Machine Learning Research</b><br><i>Edward Raff (Booz Allen Hamilton)</i></p>

      <p><b>Scalable Global Optimization via Local Bayesian Optimization</b><br><i>David Eriksson (Uber AI) &middot; Matthias Poloczek (Uber AI) &middot; Jacob Gardner (Uber AI Labs) &middot; Ryan Turner (Uber AI Labs) &middot; Michael  Pearce (Warwick University)</i></p>

      <p><b>Time-series Generative Adversarial Networks</b><br><i>Jinsung Yoon (University of California, Los Angeles) &middot; Daniel Jarrett (University of Cambridge) &middot; M Van Der Schaar (University of California, Los Angeles)</i></p>

      <p><b>On Accelerating Training of Transformer-Based Language Models</b><br><i>Qian Yang (Duke University) &middot; Zhouyuan Huo (University of Pittsburgh) &middot; Wenlin Wang (Duke Univeristy) &middot; Lawrence Carin (Duke University)</i></p>

      <p><b>A Refined Margin Distribution Analysis for Forest Representation Learning</b><br><i>Shen-Huan Lyu (Nanjing University) &middot; Liang Yang (Nanjing University) &middot; Zhi-Hua Zhou (Nanjing University)</i></p>

      <p><b>Robustness to Adversarial Perturbations in Learning from Incomplete Data</b><br><i>Amir Najafi (Sharif University of Technology) &middot; Shin-ichi Maeda (Preferred Networks) &middot; Masanori Koyama (Preferred Networks Inc. ) &middot; Takeru Miyato (Preferred Networks, Inc.)</i></p>

      <p><b>Exploring Unexplored Tensor Decompositions for Convolutional Neural Networks</b><br><i>Kohei Hayashi (Preferred Networks) &middot; Taiki Yamaguchi (The University of Tokyo) &middot; Yohei Sugawara (Preferred Networks, Inc.) &middot; Shin-ichi Maeda (Preferred Networks)</i></p>

      <p><b>An Adaptive Empirical  Bayesian Method for Sparse Deep Learning</b><br><i>Wei Deng (Purdue University) &middot; Xiao Zhang (Purdue University) &middot; Faming Liang (Purdue University) &middot; Guang Lin (Purdue University)</i></p>

      <p><b>Adaptive Influence Maximization with Myopic Feedback</b><br><i>Binghui Peng (Tsinghua University) &middot; Wei Chen (Microsoft Research)</i></p>

      <p><b>Focused Quantization for Sparse CNNs</b><br><i>Yiren Zhao (University of Cambridge) &middot; Xitong Gao (Shenzhen Institutes of Advanced Technology,Chinese Academy of Sciences) &middot; Daniel Bates (University of Cambridge) &middot; Robert Mullins (University of Cambridge) &middot; Cheng-Zhong Xu (University of Macau)</i></p>

      <p><b>Quantum Embedding of Knowledge for Reasoning</b><br><i>Dinesh Garg (IBM Research AI, India) &middot; Shajith Ikbal Mohamed (IBM Research AI, India) &middot; Santosh Srivastava (IBM Research AI) &middot; Harit Vishwakarma (IBM Research AI) &middot; Hima Karanam (IBM Research AI) &middot; L Venkat  Subramaniam (IBM India Research Lab)</i></p>

      <p><b>Optimal Best Markovian Arm Identification with Fixed Confidence</b><br><i>Vrettos Moulos (UC Berkeley)</i></p>

      <p><b>Limiting Extrapolation in Linear Approximate Value Iteration</b><br><i>Andrea Zanette (Stanford University) &middot; Alessandro Lazaric (Facebook Artificial Intelligence Research) &middot; Mykel J Kochenderfer (Stanford University) &middot; Emma Brunskill (Stanford University)</i></p>

      <p><b>Almost Horizon-Free Structure-Aware Best Policy Identification with a Generative Model</b><br><i>Andrea Zanette (Stanford University) &middot; Mykel J Kochenderfer (Stanford University) &middot; Emma Brunskill (Stanford University)</i></p>

      <p><b>Invertible Convolutional Flow</b><br><i>Mahdi Karami (University of Alberta) &middot; Dale Schuurmans (Google) &middot; Jascha Sohl-Dickstein (Google Brain) &middot; Laurent Dinh (Google Research) &middot; Daniel Duckworth (Google Brain)</i></p>

      <p><b>A Latent Variational Framework for Stochastic Optimization</b><br><i>Philippe Casgrain (University of Toronto)</i></p>

      <p><b>Topology-Preserving Deep Image Segmentation</b><br><i>Xiaoling Hu (Stony Brook University) &middot; Fuxin Li (Oregon State University) &middot; Dimitris Samaras (Stony Brook University) &middot; Chao Chen (Stony Brook University)</i></p>

      <p><b>Connective Cognition Network for Directional Visual Commonsense Reasoning</b><br><i>Aming Wu (Tianjin University) &middot; Linchao Zhu (University of Sydney, Technology) &middot; Yahong Han (Tianjin University) &middot; Yi Yang (UTS)</i></p>

      <p><b>Online Markov Decoding: Lower Bounds and Near-Optimal Approximation Algorithms</b><br><i>Vikas Garg (MIT) &middot; Tamar Pichkhadze (MIT)</i></p>

      <p><b>A Meta-MDP Approach to Exploration for Lifelong Reinforcement Learning</b><br><i>Francisco Garcia (University of Massachusetts - Amherst) &middot; Philip Thomas (University of Massachusetts Amherst)</i></p>

      <p><b>Push-pull Feedback Implements Hierarchical Information Retrieval Efficiently</b><br><i>Xiao Liu (Peking University) &middot; Xiaolong Zou (Peking University) &middot; Zilong Ji (Beijing Normal University) &middot; Gengshuo Tian (Beijing Normal University) &middot; Yuanyuan Mi (Weizmann Institute of Science) &middot; Tiejun Huang (Peking University) &middot; K. Y. Michael Wong (Department of Physics, Hong Kong University of Science and Technology) &middot; Si Wu (Peking University)</i></p>

      <p><b>Learning Disentangled Representations for Recommendation</b><br><i>Jianxin Ma (Tsinghua University) &middot; Chang Zhou (Alibaba Group) &middot; Peng Cui (Tsinghua University) &middot; Hongxia Yang (Alibaba Group) &middot; Wenwu Zhu (Tsinghua University)</i></p>

      <p><b>Graph Neural Tangent Kernel: Fusing Graph Neural Networks with Graph Kernels</b><br><i>Simon Du (Carnegie Mellon University) &middot; Kangcheng Hou (Zhejiang University) &middot; Ruslan Salakhutdinov (Carnegie Mellon University) &middot; Barnabas Poczos (Carnegie Mellon University) &middot; Ruosong Wang (Carnegie Mellon University) &middot; Keyulu Xu (MIT)</i></p>

      <p><b>In-Place Near Zero-Cost Memory Protection for DNN</b><br><i>Hui Guan (North Carolina State University) &middot; Lin Ning (NCSU) &middot; Zhen Lin (NCSU) &middot; Xipeng Shen (North Carolina State University) &middot; Huiyang Zhou (NCSU) &middot;  Seung-Hwan Lim (Oak Ridge National Laboratory)</i></p>

      <p><b>Acceleration via Symplectic Discretization of High-Resolution Differential Equations</b><br><i>Bin Shi (UC Berkeley) &middot; Simon Du (Carnegie Mellon University) &middot; Weijie Su (University of Pennsylvania) &middot; Michael Jordan (UC Berkeley)</i></p>

      <p><b>XLNet: Generalized Autoregressive Pretraining for Language Understanding</b><br><i>Zhilin Yang (Tsinghua University) &middot; Zihang Dai (Carnegie Mellon University) &middot; Yiming Yang (CMU) &middot; Jaime Carbonell (CMU) &middot; Ruslan Salakhutdinov (Carnegie Mellon University) &middot; Quoc V Le (Google)</i></p>

      <p><b>Comparison Against Task Driven Artificial Neural Networks Reveals Functional Properties in Mouse Visual Cortex</b><br><i>Jianghong Shi (University of Washington) &middot; Eric Shea-Brown (University of Washington) &middot; Michael Buice (Allen Institute for Brain Science)</i></p>

      <p><b>Mixtape: Breaking the Softmax Bottleneck Efficiently</b><br><i>Zhilin Yang (Tsinghua University) &middot; Thang Luong (Google) &middot; Ruslan Salakhutdinov (Carnegie Mellon University) &middot; Quoc V Le (Google)</i></p>

      <p><b>Variance Reduced Policy Evaluation with Smooth Function Approximation</b><br><i>Hoi-To Wai (Chinese University of Hong Kong) &middot; Mingyi Hong (University of Minnesota) &middot; Zhuoran Yang (Princeton University) &middot; Zhaoran Wang (Northwestern University) &middot; Kexin Tang (University of Minnesota)</i></p>

      <p><b>Learning GANs and Ensembles Using Discrepancy</b><br><i>Ben Adlam (Google) &middot; Corinna Cortes (Google Research) &middot; Mehryar Mohri (Courant Inst. of Math. Sciences & Google Research) &middot; Ningshan Zhang (NYU)</i></p>

      <p><b>Co-Generation with GANs using AIS based HMC</b><br><i>Tiantian Fang (University of Illinois Urbana-Champaign) &middot; Alexander Schwing (University of Illinois at Urbana-Champaign)</i></p>

      <p><b>AttentionXML: Label Tree-based Attention-Aware Deep Model for High-Performance Extreme Multi-Label Text Classification</b><br><i>Ronghui You (Fudan University) &middot; Zihan Zhang (Fudan University) &middot; Ziye Wang (Fudan University) &middot; Suyang Dai (Fudan University) &middot; Hiroshi Mamitsuka (Kyoto University) &middot; Shanfeng Zhu (Fudan University)</i></p>

      <p><b>Addressing Sample Complexity in Visual Tasks Using HER and Hallucinatory GANs</b><br><i>Himanshu Sahni (Georgia Institute of Technology) &middot; Toby Buckley (Offworld Inc.) &middot; Pieter Abbeel (University of California, Berkley & OpenAI) &middot; Ilya Kuzovkin (Offworld Inc.)</i></p>

      <p><b>Abstract Reasoning with Distracting Features</b><br><i>Kecheng Zheng (University of Science and Technology of China) &middot; Zheng-Jun Zha (University of Science and Technology of China) &middot; Wei Wei (Google AI)</i></p>

      <p><b>Generalized Block-Diagonal Structure Pursuit: Learning Soft Latent Task Assignment against Negative Transfer</b><br><i>Zhiyong Yang (SKLOIS, Institute of Information Engineering, Chinese Academy of Sciences; SCS, University of Chinese Academy of Sciences) &middot; Qianqian Xu (Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences) &middot; Yangbangyan Jiang (SKLOIS, Institute of Information Engineering, Chinese Academy of Sciences; SCS, University of Chinese Academy of Sciences) &middot; Xiaochun Cao (Chinese Academy of Sciences) &middot; Qingming Huang (University of Chinese Academy of Sciences)</i></p>

      <p><b>Adversarial Training and Robustness for Multiple Perturbations</b><br><i>Florian Tramer (Stanford University) &middot; Dan Boneh (Stanford University)</i></p>

      <p><b>Doubly-Robust Lasso Bandit</b><br><i>Gi-Soo Kim (Seoul National University) &middot; Myunghee Cho Paik (Seoul National University)</i></p>

      <p><b>DM2C: Deep Mixed-Modal Clustering</b><br><i>Yangbangyan Jiang (SKLOIS, Institute of Information Engineering, Chinese Academy of Sciences; SCS, University of Chinese Academy of Sciences) &middot; Qianqian Xu (Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences) &middot; Zhiyong Yang (SKLOIS, Institute of Information Engineering, Chinese Academy of Sciences; SCS, University of Chinese Academy of Sciences) &middot; Xiaochun Cao (Chinese Academy of Sciences) &middot; Qingming Huang (University of Chinese Academy of Sciences)</i></p>

      <p><b>MaCow: Masked Convolutional Generative Flow</b><br><i>Xuezhe Ma (Carnegie Mellon University) &middot; Xiang Kong (Carnegie Mellon University) &middot; Shanghang Zhang (Carnegie Mellon University) &middot; Eduard Hovy (Carnegie Mellon University)</i></p>

      <p><b>Learning by Abstraction: The Neural State Machine for Visual Reasoning</b><br><i>Drew Hudson (Stanford) &middot; Christopher Manning (Stanford University)</i></p>

      <p><b>Adaptive Gradient-Based Meta-Learning Methods</b><br><i>Mikhail Khodak (CMU) &middot; Maria-Florina Balcan (Carnegie Mellon University) &middot; Ameet Talwalkar (CMU)</i></p>

      <p><b>Equipping Experts/Bandits with Long-term Memory</b><br><i>Kai Zheng (Peking University) &middot; Haipeng Luo (University of Southern California) &middot; Ilias Diakonikolas (USC) &middot; Liwei Wang (Peking University)</i></p>

      <p><b>A Regularized Approach to Sparse Optimal Policy in Reinforcement Learning</b><br><i>Wenhao Yang (Peking University) &middot; Xiang Li (Peking University) &middot; Zhihua Zhang (Peking University)</i></p>

      <p><b>Scalable inference of topic evolution via models for latent geometric structures</b><br><i>Mikhail Yurochkin (IBM Research, MIT-IBM Watson AI Lab) &middot; Zhiwei Fan (University of Wisconsin-Madison) &middot; Aritra Guha (University of Michigan) &middot; Paraschos Koutris (University of Wisconsin-Madison) &middot; XuanLong Nguyen (University of Michigan)</i></p>

      <p><b>Effective End-to-end Unsupervised Outlier Detection via Inlier Priority of Discriminative Network</b><br><i>Siqi Wang (National University of Defense Technology) &middot; Yijie Zeng (Nanyang Technological University) &middot; Xinwang Liu (National University of Defense Technology) &middot; En Zhu (National University of Defense Technology) &middot; Jianping Yin (Dongguan University of Technology) &middot; Chuanfu Xu (National University of Defense Technology) &middot; Marius Kloft (TU Kaiserslautern)</i></p>

      <p><b>Deep Active Learning with a Neural Architecture Search</b><br><i>Yonatan Geifman (Technion) &middot; Ran El-Yaniv (Technion)</i></p>

      <p><b>Efficiently escaping saddle points on manifolds</b><br><i>Christopher Criscitiello (Princeton University) &middot; Nicolas Boumal (Princeton University)</i></p>

      <p><b>AutoAssist: A Framework to Accelerate Training of Deep Neural Networks</b><br><i>Jiong Zhang (University of Texas at Austin) &middot; Hsiang-Fu Yu (Amazon) &middot; Inderjit S Dhillon (UT Austin & Amazon)</i></p>

      <p><b>DFNets: Spectral CNNs for Graphs with Feedback-looped Filters</b><br><i>W. O. K. Asiri Suranga Wijesinghe (The Australian National University) &middot; Qing Wang (Australian National University)</i></p>

      <p><b>Learning Dynamics of Attention: Human Prior for Interpretable Machine Reasoning</b><br><i>Wonjae Kim (Kakao Corporation) &middot; Yoonho Lee (Kakao Corporation)</i></p>

      <p><b>Comparing Unsupervised Word Translation Methods Step by Step</b><br><i>Mareike Hartmann (University of Copenhagen) &middot; Yova  Kementchedjhieva (University of Copenhagen) &middot; Anders Søgaard (University of Copenhagen)</i></p>

      <p><b>Learning from Crap Data via Generation</b><br><i>Tianyu Guo (Peking University) &middot; Chang Xu (University of Sydney) &middot; Boxin Shi (Peking University) &middot; Chao Xu (Peking University) &middot; Dacheng Tao (University of Sydney)</i></p>

      <p><b>Constrained deep neural network architecture search for IoT devices accounting hardware calibration</b><br><i>Florian Scheidegger (IBM Research -- Zurich) &middot; Luca Benini (ETHZ, University of Bologna ) &middot; Costas Bekas (IBM Research GmbH) &middot; A. Cristiano I. Malossi (IBM Research - Zurich)</i></p>

      <p><b>Quantum Entropy Scoring for Fast Robust Mean Estimation and Improved Outlier Detection</b><br><i>Yihe Dong (Microsoft Research) &middot; Sam Hopkins (UC Berkeley) &middot; Jerry Li (Microsoft)</i></p>

      <p><b>Iterative Least Trimmed Squares for Mixed Linear Regression</b><br><i>Yanyao Shen (UT Austin) &middot; Sujay Sanghavi (UT-Austin)</i></p>

      <p><b>Dynamic Ensemble Modeling Approach to Nonstationary Neural Decoding in Brain-Computer Interfaces</b><br><i>Yu Qi (Zhejiang University) &middot; Bin Liu (Nanjing University of Posts and Telecommunications) &middot; Yueming Wang (Zhejiang University) &middot; Gang Pan (Zhejiang University)</i></p>

      <p><b>Divergence-Augmented Policy Optimization</b><br><i>Qing Wang (Tencent AI Lab) &middot; Yingru Li (The Chinese University of Hong Kong, Shenzhen) &middot; Jiechao Xiong (Tencent AI Lab) &middot; Tong Zhang (Tencent AI Lab)</i></p>

      <p><b>Intrinsic dimension of data representations in deep neural networks</b><br><i>Alessio Ansuini (International School for Advanced Studies (SISSA)) &middot; Alessandro Laio (International School for Advanced Studies (SISSA)) &middot; Jakob H Macke (Technical University of Munich, Munich, Germany) &middot; Davide Zoccolan (Visual Neuroscience Lab, International School for Advanced Studies (SISSA))</i></p>

      <p><b>Towards a Zero-One Law for Column Subset Selection</b><br><i>Zhao Song (University of Washington) &middot; David Woodruff (Carnegie Mellon University) &middot; Peilin Zhong (Columbia University)</i></p>

      <p><b>Compositional De-Attention Networks</b><br><i>Yi Tay (Nanyang Technological University) &middot; Anh Tuan Luu (MIT CSAIL) &middot; Aston Zhang (Amazon AI) &middot; Shuohang Wang (Singapore Management University) &middot; Siu Cheung Hui (Nanyang Technological University)</i></p>

      <p><b>Dual Adversarial Semantics-Consistent Network for Generalized Zero-Shot Learning</b><br><i>Jian Ni (University of Science and Technology of China) &middot; Shanghang Zhang (Carnegie Mellon University) &middot; Haiyong Xie (University of Science and Technology of China)</i></p>

      <p><b>Learning and Generalization in Overparameterized Neural Networks, Going Beyond Two Layers</b><br><i>Zeyuan Allen-Zhu (Microsoft Research) &middot; Yuanzhi Li (Princeton) &middot; Yingyu Liang (University of Wisconsin Madison)</i></p>

      <p><b>Mining GOLD Samples for Conditional GANs</b><br><i>Sangwoo Mo (KAIST) &middot; Chiheon Kim (Kakao Brain) &middot; Sungwoong  Kim (Kakao Brain) &middot; Minsu Cho (POSTECH) &middot; Jinwoo Shin (KAIST; AITRICS)</i></p>

      <p><b>Deep Model Transferability from Attribution Maps</b><br><i>Jie Song (Zhejiang University) &middot; Yixin Chen (Zhejiang University) &middot; Xinchao Wang (Stevens Institute of Technology) &middot; Chengchao Shen (Zhejiang University) &middot; Mingli Song (Zhejiang University)</i></p>

      <p><b>Fully Parameterized Quantile Function for Distributional Reinforcement Learning</b><br><i>Derek C Yang (UC San Diego) &middot; Li Zhao (Microsoft Research) &middot; Zichuan Lin (Tsinghua University) &middot; Tao Qin (Microsoft Research) &middot; Jiang Bian (Microsoft) &middot; Tie-Yan Liu (Microsoft Research Asia)</i></p>

      <p><b>Direct Optimization through $\arg \max$ for Discrete Variational Auto-Encoder</b><br><i>Guy Lorberbom (Technion) &middot; Tommi Jaakkola (MIT) &middot; Andreea Gane (Google AI) &middot; Tamir Hazan (Technion)</i></p>

      <p><b>Distributional Reward Decomposition for Reinforcement Learning</b><br><i>Zichuan Lin (Tsinghua University) &middot; Li Zhao (Microsoft Research) &middot; Derek C Yang (UC San Diego) &middot; Tao Qin (Microsoft Research) &middot; Tie-Yan Liu (Microsoft Research Asia) &middot; Guangwen Yang (Tsinghua University)</i></p>

      <p><b>L_DMI: A Novel Information-theoretic Loss Function for Training Deep Nets Robust to Label Noise</b><br><i>Yilun Xu (Peking University) &middot; Peng Cao (Peking University) &middot; Yuqing Kong (Peking University) &middot; Yizhou Wang (Peking University)</i></p>

      <p><b>Convergence Guarantees for Adaptive Bayesian Quadrature Methods</b><br><i>Motonobu Kanagawa (EURECOM) &middot; Philipp Hennig (University of Tübingen and MPI for Intelligent Systems Tübingen)</i></p>

      <p><b>Progressive Augmentation of GANs</b><br><i>Dan Zhang (Bosch Center for Artificial Intelligence) &middot; Anna Khoreva (Bosch Center for AI)</i></p>

      <p><b>UniXGrad: A Universal, Adaptive Algorithm with Optimal Guarantees for Constrained Optimization</b><br><i>Ali Kavis (EPFL) &middot; Yehuda Kfir Levy (ETH) &middot; Francis Bach (INRIA - Ecole Normale Superieure) &middot; Volkan Cevher (EPFL)</i></p>

      <p><b>Meta-Surrogate Benchmarking for Hyperparameter Optimization</b><br><i>Aaron Klein (Amazon Berlin) &middot; Zhenwen Dai (Spotify) &middot; Frank Hutter (University of Freiburg) &middot; Neil Lawrence (Amazon) &middot; Javier Gonzalez (Amazon)</i></p>

      <p><b>Learning to Perform Local Rewriting for Combinatorial Optimization</b><br><i>Xinyun Chen (UC Berkeley) &middot; Yuandong Tian (Facebook AI Research)</i></p>

      <p><b>Anti-efficient encoding in emergent communication</b><br><i>Rahma Chaabouni (LSCP-FAIR) &middot; Eugene Kharitonov (Facebook AI) &middot; Emmanuel Dupoux (Ecole des Hautes Etudes en Sciences Sociales) &middot; Marco Baroni (University of Trento)</i></p>

      <p><b>Singleshot : a scalable Tucker tensor decomposition</b><br><i>Abraham Traore () &middot; Maxime Berar (Université de Rouen) &middot; Alain Rakotomamonjy (Université de Rouen Normandie   Criteo AI Lab)</i></p>

      <p><b>Neural Machine Translation with Soft Prototype</b><br><i>Yiren Wang (University of Illinois at Urbana-Champaign) &middot; Yingce Xia (Microsoft Research Asia) &middot; Fei Tian (Microsoft Research) &middot; Fei Gao (University of Chinese Academy of Sciences) &middot; Tao Qin (Microsoft Research) &middot; Cheng Xiang  Zhai (University of Illinois at Urbana-Champaign) &middot; Tie-Yan Liu (Microsoft Research)</i></p>

      <p><b>Reliable training and estimation of variance networks</b><br><i>Nicki Skafte Detlefsen (Technical University of Denmark) &middot; Martin Jørgensen (Technical University of Denmark) &middot; Søren Hauberg (Technical University of Denmark)</i></p>

      <p><b>On the Statistical Properties of Multilabel Learning</b><br><i>Weiwei Liu (Wuhan University)</i></p>

      <p><b>Bayesian Learning of Sum-Product Networks</b><br><i>Martin Trapp (Graz University of Technology) &middot; Robert Peharz (University of Cambridge) &middot; Hong Ge (University of Cambridge) &middot; Franz Pernkopf (Signal Processing and Speech Communication Laboratory, Graz, Austria) &middot; Zoubin Ghahramani (Uber and University of Cambridge)</i></p>

      <p><b>Bayesian Batch Active Learning as Sparse Subset Approximation</b><br><i>Robert Pinsler (University of Cambridge) &middot; Jonathan Gordon (University of Cambridge) &middot; Eric Nalisnick (University of Cambridge) &middot; José Miguel Hernández-Lobato (University of Cambridge)</i></p>

      <p><b>Optimal Sparsity-Sensitive Bounds for  Distributed Mean Estimation</b><br><i>zengfeng Huang (Fudan University) &middot; Ziyue Huang (HKUST) &middot; Yilei WANG (The Hong Kong University of Science and Technology) &middot; Ke Yi (" Hong Kong University of Science and Technology, Hong Kong")</i></p>

      <p><b>Global Sparse Momentum SGD for Pruning Very Deep Neural Networks</b><br><i>Xiaohan Ding (Tsinghua University) &middot; guiguang ding (Tsinghua University, China) &middot; Xiangxin Zhou (Tsinghua University) &middot; Yuchen Guo (Tsinghua University) &middot; Jungong Han (Lancaster University) &middot; Ji Liu (University of Rochester, Tencent AI lab)</i></p>

      <p><b>Variational Bayesian Decision-making for Continuous Utilities</b><br><i>Tomasz Kuśmierczyk (University of Helsinki) &middot; Joseph Sakaya (University of Helsinki) &middot; Arto Klami (University of Helsinki)</i></p>

      <p><b>The Normalization Method for Alleviating Pathological Sharpness in Wide Neural Networks</b><br><i>Ryo Karakida (National Institute of Advanced Industrial Science and Technology) &middot; Shotaro Akaho (AIST) &middot; Shun-ichi Amari (RIKEN)</i></p>

      <p><b>Single-Model Uncertainties for Deep Learning</b><br><i>Natasa Tagasovska (University of Lausanne) &middot; David Lopez-Paz (Facebook AI Research)</i></p>

      <p><b>Is Deeper Better only when Shallow is Good?</b><br><i>Eran Malach (Hebrew University Jerusalem Israel) &middot; Shai Shalev-Shwartz (Mobileye & HUJI)</i></p>

      <p><b>Wasserstein Weisfeiler-Lehman Graph Kernels</b><br><i>Matteo Togninalli (ETH Zürich) &middot; Elisabetta Ghisu (ETH Zurich) &middot; Felipe Llinares-Lopez (ETH Zürich) &middot; Bastian Rieck (MLCB, D-BSSE, ETH Zurich) &middot; Karsten Borgwardt (ETH Zurich)</i></p>

      <p><b>Domain Generalization via Model-Agnostic Learning of Semantic Features</b><br><i>Qi Dou (Imperial College London) &middot; Daniel Coelho de Castro (Imperial College London) &middot; Konstantinos Kamnitsas (Imperial College London) &middot; Ben Glocker (Imperial College London)</i></p>

      <p><b>Grid Saliency for Context Explanations of Semantic Segmentation</b><br><i>Lukas Hoyer (Bosch Center for Artificial Intelligence) &middot; Mauricio Munoz (Bosch Center for Artificial Intelligence) &middot; Prateek Katiyar (Bosch Center for Artificial Intelligence) &middot; Anna Khoreva (Bosch Center for AI) &middot; Volker Fischer (Robert Bosch GmbH, Bosch Center for Artificial Intelligence)</i></p>

      <p><b>First-order methods almost always avoid saddle points: The case of Vanishing step-sizes</b><br><i>Ioannis Panageas (SUTD) &middot; Georgios Piliouras (Singapore University of Technology and Design) &middot; Xiao Wang (Singapore University of Technology and Design)</i></p>

      <p><b>Maximum Mean Discrepancy Gradient Flow</b><br><i>Michael Arbel (UCL) &middot; Anna Korba (UCL) &middot; Adil SALIM (KAUST) &middot; Arthur Gretton (Gatsby Unit, UCL)</i></p>

      <p><b>Oblivious Sampling Algorithms for Private Data Analysis</b><br><i>Olga Ohrimenko (Microsoft Research) &middot; Sajin Sasy (University of Waterloo)</i></p>

      <p><b>Semi-supervisedly Co-embedding Attributed Networks</b><br><i>Zai Qiao Meng (University of Glasgow) &middot; Shangsong Liang (Sun Yat-sen University) &middot; Jinyuan Fang (Sun Yat-sen University) &middot; Teng Xiao (Sun Yat-sen University)</i></p>

      <p><b>From voxels to pixels and back: Self-supervision in natural-image reconstruction from fMRI</b><br><i>Roman Beliy (weizmann institute) &middot; Guy Gaziv (Weizmann Institute of Science) &middot; Assaf Hoogi (Weizmann Institute) &middot; Francesca Strappini (Weizmann Institute of Science) &middot; Tal Golan (Columbia University) &middot; Michal Irani (The Weizmann Institute of Science)</i></p>

      <p><b>Copulas as High-Dimensional Generative Models: Vine Copula Autoencoders</b><br><i>Natasa Tagasovska (University of Lausanne) &middot; Damien Ackerer (Swissquote) &middot; Thibault Vatter (Columbia University)</i></p>

      <p><b>Nonstochastic Multiarmed Bandits with Unrestricted Delays</b><br><i>Tobias Sommer Thune (University of Copenhagen) &middot; Nicolò Cesa-Bianchi (Università degli Studi di Milano) &middot; Yevgeny Seldin (University of Copenhagen)</i></p>

      <p><b>BIVA: A Very Deep Hierarchy of Latent Variables for Generative Modeling</b><br><i>Lars Maaløe (Corti) &middot; Marco Fraccaro (Unumed) &middot; Valentin Liévin (DTU) &middot; Ole Winther (Technical University of Denmark)</i></p>

      <p><b>Code Generation as Dual Task of Code Summarization</b><br><i>Bolin Wei (Peking University) &middot; Ge Li (Peking University) &middot; Xin Xia (Monash University) &middot; Zhiyi Fu (Key Lab of High Confidence Software Technologies (Peking University), Ministry o) &middot; Zhi Jin (Key Lab of High Confidence Software Technologies (Peking University), Ministry o)</i></p>

      <p><b>Diffeomorphic Temporal Alignment Networks</b><br><i>Ron Shapira weber (Ben Gurion University) &middot; Matan Eyal (Ben Gurion University) &middot; Nicki Skafte Detlefsen (Technical University of Denmark) &middot; Oren Shriki (Ben-Gurion University of the Negev) &middot; Oren Freifeld (Ben-Gurion University)</i></p>

      <p><b>Weakly Supervised Instance Segmentation using the Bounding Box Tightness Prior</b><br><i>Cheng-Chun  Hsu (Academia Sinica) &middot; Kuang-Jui Hsu (Qualcomm) &middot; Chung-Chi Tsai  (Qualcomm) &middot; Yen-Yu Lin (National Chiao Tung University) &middot; Yung-Yu Chuang (National Taiwan University)</i></p>

      <p><b>On the Power and Limitations of Random Features for Understanding Neural Networks</b><br><i>Gilad Yehudai (Weizmann Institute of Science) &middot; Ohad Shamir (Weizmann Institute of Science)</i></p>

      <p><b>Efficient Pure Exploration in Adaptive Round model</b><br><i>tianyuan jin (University of Science and Technology of China) &middot; Jieming SHI (NATIONAL UNIVERSITY OF SINGAPORE) &middot; Xiaokui Xiao (National University of Singapore) &middot; Enhong Chen (University of Science and Technology of China)</i></p>

      <p><b>Multi-objects Generation with Amortized Structural Regularization</b><br><i>Taufik Xu (Tsinghua University) &middot; Chongxuan LI (Tsinghua University) &middot; Jun Zhu (Tsinghua University) &middot; Bo Zhang (Tsinghua University)</i></p>

      <p><b>Neural Shuffle-Exchange Networks - Sequence Processing in O(n log n) Time</b><br><i>Karlis Freivalds (Institute of Mathematics and Computer Science) &middot; Emīls Ozoliņš (Institute of Mathematics and Computer Science) &middot; Agris Šostaks (Institute of Mathematics and Computer Science)</i></p>

      <p><b>DetNAS: Backbone Search for Object Detection</b><br><i>Yukang Chen (Institute of Automation, Chinese Academy of Sciences) &middot; Tong Yang (Megvii Inc.) &middot; Xiangyu Zhang (Megvii Inc (Face++)) &middot; GAOFENG MENG (Institute of Automation, Chinese Academy of Sciences) &middot; Xinyu Xiao (National Laboratory of Pattern recognition (NLPR),  Institute of Automation of Chinese Academy of Sciences (CASIA)) &middot; Jian Sun (Megvii, Face++)</i></p>

      <p><b>Stochastic Proximal Langevin Algorithm: Potential Splitting and Nonasymptotic Rates</b><br><i>Adil SALIM (KAUST) &middot; Dmitry Koralev (KAUST) &middot; Peter Richtarik (KAUST)</i></p>

      <p><b>Fast AutoAugment</b><br><i>Sungbin Lim (Kakao Brain) &middot; Ildoo Kim (Kakao Brain) &middot; Taesup Kim (Mila / Kakao Brain) &middot; Chiheon Kim (Kakao Brain) &middot; Sungwoong  Kim (Kakao Brain)</i></p>

      <p><b>On the Convergence Rate of Training Recurrent Neural Networks in the Overparameterized Regime</b><br><i>Zeyuan Allen-Zhu (Microsoft Research) &middot; Yuanzhi Li (Princeton) &middot; Zhao Song (University of Washington)</i></p>

      <p><b>Interval timing in deep reinforcement learning agents</b><br><i>Ben Deverett (DeepMind) &middot; Ryan Faulkner (Deepmind) &middot; Meire Fortunato (DeepMind) &middot; Gregory Wayne (Google DeepMind) &middot; Joel Leibo (DeepMind)</i></p>

      <p><b>Graph-based Discriminators: Sample Complexity and Expressiveness</b><br><i>Roi Livni (Tel Aviv University) &middot; Yishay Mansour (Tel Aviv University / Google)</i></p>

      <p><b>Large Scale Structure of Neural Network Loss Landscapes</b><br><i>Stanislav Fort (Stanford University) &middot; Stanislaw Jastrzebski (New York University)</i></p>

      <p><b>Learning Nonsymmetric Determinantal Point Processes</b><br><i>Mike Gartrell (Criteo AI Lab) &middot; Victor-Emmanuel Brunel (ENSAE ParisTech) &middot; Elvis Dohmatob (Criteo) &middot; Syrine Krichene (Google)</i></p>

      <p><b>Hypothesis Set Stability and Generalization</b><br><i>Dylan Foster (MIT) &middot; Spencer Greenberg (Spark Wave) &middot; Satyen Kale (Google) &middot; Haipeng Luo (University of Southern California) &middot; Mehryar Mohri (Courant Inst. of Math. Sciences & Google Research) &middot; Karthik Sridharan (Cornell University)</i></p>

      <p><b>Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds</b><br><i>Bo Yang (University of Oxford) &middot; Jianan Wang (DeepMind) &middot; Ronald Clark (Imperial College London) &middot; Qingyong Hu (University of Oxford) &middot; Sen Wang (Heriot-Watt University) &middot; Andrew Markham (University of Oxford) &middot; Niki Trigoni (University of Oxford)</i></p>

      <p><b>Precision-Recall Balanced Topic Modelling</b><br><i>Seppo Virtanen (Imperial College London) &middot; Mark Girolami (Imperial College London)</i></p>

      <p><b>Learning Sparse Distributions using Iterative Hard Thresholding</b><br><i>Yibo Zhang (Illinois) &middot; Rajiv Khanna (University of California at Berkeley) &middot; Anastasios Kyrillidis (Rice University ) &middot; Oluwasanmi Koyejo (UIUC)</i></p>

      <p><b>Discriminative Topic Modeling with Logistic LDA</b><br><i>Iryna Korshunova (Ghent University) &middot; Hanchen Xiong (Twitter) &middot; Mateusz Fedoryszak (Twitter) &middot; Lucas Theis (Twitter)</i></p>

      <p><b>Quantum Wasserstein Generative Adversarial Networks</b><br><i>Shouvanik Chakrabarti (University of Maryland) &middot; Huang Yiming (University of Maryland & University of Electronic Science and Technology of China) &middot; Tongyang Li (University of Maryland) &middot; Soheil Feizi (University of Maryland, College Park) &middot; Xiaodi Wu (University of Maryland)</i></p>

      <p><b>Blow: a single-scale hyperconditioned flow for non-parallel raw-audio voice conversion</b><br><i>Joan Serrà (Telefónica Research) &middot; Santiago Pascual (Universitat Politècnica de Catalunya) &middot; Carlos Segura Perales (Telefónica Research)</i></p>

      <p><b>Hyperparameter Learning via Distributional Transfer</b><br><i>Ho Chung Law (University of Oxford) &middot; Peilin Zhao (Tencent AI Lab) &middot; Lucian Chan (University of Oxford) &middot; Junzhou Huang (University of Texas at Arlington / Tencent AI Lab) &middot; Dino Sejdinovic (University of Oxford)</i></p>

      <p><b>Discriminator optimal transport</b><br><i>Akinori Tanaka (RIKEN)</i></p>

      <p><b>High-dimensional multivariate forecasting with low-rank Gaussian Copula Processes</b><br><i>David Salinas (Amazon) &middot; Michael Bohlke-Schneider (Amazon) &middot; Laurent Callot (Amazon) &middot; Jan Gasthaus (Amazon.com) &middot; Roberto Medico (Amazon AWS)</i></p>

      <p><b>Are Anchor Points Really Indispensable in Label-Noise Learning?</b><br><i>Xiaobo Xia (The University of Sydney / Xidian University) &middot; Tongliang Liu (The University of Sydney) &middot; Nannan Wang (Xidian University) &middot; Bo Han (RIKEN) &middot; Chen Gong (Nanjing University of Science and Technology) &middot; Gang Niu (RIKEN) &middot; Masashi Sugiyama (RIKEN / University of Tokyo)</i></p>

      <p><b>Aligning Visual Regions and Textual Concepts for Semantic-Grounded Image Representations</b><br><i>Fenglin Liu (Peking University) &middot; Yuanxin Liu (Institute of Information Engineering, Chinese Academy of Sciences) &middot; Xuancheng Ren (Peking University) &middot; Xiaodong He (JD AI research) &middot; Kai Lei (peking university) &middot; Xu Sun (Peking University)</i></p>

      <p><b>Differentiable Sorting using Optimal Transport: The Sinkhorn CDF and Quantile Operator</b><br><i>Marco Cuturi (Google and CREST/ENSAE) &middot; Olivier Teboul (Google Brain) &middot; Jean-Philippe Vert ()</i></p>

      <p><b>Dichotomize and Generalize: PAC-Bayesian Binary Activated Deep Neural Networks</b><br><i>Gaël Letarte (Université Laval) &middot; Pascal Germain (INRIA) &middot; Benjamin Guedj (Inria & University College London) &middot; Francois Laviolette (Université Laval)</i></p>

      <p><b>Likelihood-Free Overcomplete ICA and ApplicationsIn Causal Discovery</b><br><i>Chenwei DING (The University of Sydney) &middot; Mingming Gong (University of Melbourne) &middot; Kun Zhang (CMU) &middot; Dacheng Tao (University of Sydney)</i></p>

      <p><b>Interior-point Methods Strike Back: Solving the Wasserstein Barycenter Problem</b><br><i>DongDong Ge (Shanghai University of Finance and Economics) &middot; Haoyue Wang (Fudan University) &middot; Zikai Xiong (Fudan University) &middot; Yinyu  Ye (Standord)</i></p>

      <p><b>Beyond Vector Spaces: Compact Data Representation as Differentiable Weighted Graphs</b><br><i>Denis Mazur (Yandex) &middot; Vage Egiazarian (Skoltech) &middot; Stanislav Morozov (Yandex) &middot; Artem Babenko (Yandex)</i></p>

      <p><b>Subspace Detours: Building Transport Plans that are Optimal on Subspace Projections</b><br><i>Boris Muzellec (ENSAE, Institut Polytechnique de Paris) &middot; Marco Cuturi (Google and CREST/ENSAE)</i></p>

      <p><b>Efficient Non-Convex Stochastic Compositional Optimization Algorithm via Stochastic Recursive Gradient Descent</b><br><i>Huizhuo Yuan (Peking University) &middot; Xiangru Lian (University of Rochester) &middot; Chris Junchi Li (Tencent AI Lab) &middot; Ji Liu (University of Rochester, Tencent AI lab)</i></p>

      <p><b>On the convergence of single-call stochastic extra-gradient methods</b><br><i>Yu-Guan Hsieh (École normale supérieure, Paris) &middot; Franck Iutzeler (Univ. Grenoble Alpes) &middot; Jérôme Malick (CNRS and LJK) &middot; Panayotis Mertikopoulos (CNRS (French National Center for Scientific Research))</i></p>

      <p><b>Infra-slow brain dynamics as a marker for cognitive function and decline</b><br><i>Shagun Ajmera (Indian Institute of Science) &middot; Shreya Rajagopal (Indian Institute of Science) &middot; Razi Rehman (Indian Institute of Science) &middot; Devarajan Sridharan (Indian Institute of Science)</i></p>

      <p><b>Robust Principle Component Analysis with Adaptive Neighbors</b><br><i>Rui Zhang (Northwestern Polytechincal University) &middot; Hanghang Tong (IBM Research)</i></p>

      <p><b>High-Quality Self-Supervised Deep Image Denoising</b><br><i>Samuli Laine (NVIDIA) &middot; Tero Karras (NVIDIA) &middot; Jaakko Lehtinen (Aalto University & NVIDIA) &middot; Timo Aila (NVIDIA Research)</i></p>

      <p><b>Dynamics of stochastic gradient descent for two-layer neural networks in the teacher-student setup</b><br><i>Sebastian Goldt (Institut de Physique théorique, Paris) &middot; Madhu Advani (Harvard University) &middot; Andrew Saxe (University of Oxford) &middot; Florent Krzakala (École Normale Supérieure) &middot; Lenka Zdeborová (CEA Saclay)</i></p>

      <p><b>GIFT: Learning Transformation-Invariant Dense Visual Descriptors via Group CNNs</b><br><i>Yuan Liu (Zhejiang University) &middot; Zehong Shen (Zhejiang University) &middot; Zhixuan Lin (Zhejiang University) &middot; Sida Peng (Zhejiang University) &middot; Hujun Bao (Zhejiang University) &middot; Xiaowei Zhou (Zhejiang Univ., China)</i></p>

      <p><b>Online Prediction of Switching Graph Labelings with Cluster Specialists</b><br><i>Mark Herbster (University College London) &middot; James Robinson (UCL)</i></p>

      <p><b>Graph-Based Semi-Supervised Learning with Non-ignorable Non-response</b><br><i>Fan Zhou (Shanghai University of Finance and Economics) &middot; Tengfei Li (UNC Chapel Hill) &middot; Haibo Zhou (University of North Carolina at Chapel Hill) &middot; Hongtu Zhu (UNC Chapel Hill) &middot; Ye Jieping (DiDi Chuxing)</i></p>

      <p><b>BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning</b><br><i>Andreas Kirsch (University of Oxford) &middot; Joost van Amersfoort (University of Oxford) &middot; Yarin Gal (University of Oxford)</i></p>

      <p><b>A Mean Field Theory of Quantized Deep Networks: The Quantization-Depth Trade-Off</b><br><i>Yaniv Blumenfeld (Technion) &middot; Dar Gilboa (Columbia University) &middot; Daniel Soudry (Technion)</i></p>

      <p><b>Beyond Confidence Regions: Tight Bayesian Ambiguity Sets for Robust MDPs</b><br><i>Marek Petrik (University of New Hampshire) &middot; Reazul Hasan Russel (University of New Hampshire)</i></p>

      <p><b>Cross-lingual Language Model Pretraining</b><br><i>Alexis CONNEAU (Facebook) &middot; Guillaume Lample (Facebook AI Research)</i></p>

      <p><b>Approximate Bayesian Inference for a Mechanistic Model of Vesicle Release at a Ribbon Synapse</b><br><i>Cornelius Schröder (University of Tübingen) &middot; Ben James  (University of Sussex) &middot; Leon Lagnado (University of Sussex) &middot; Philipp Berens (University of Tübingen)</i></p>

      <p><b>Updates of Equilibrium Prop Match Gradients of Backprop Through Time in an RNN with Static Input</b><br><i>Maxence Ernoult (Université Paris Sud) &middot; Benjamin Scellier () &middot; Yoshua Bengio (Mila) &middot; Damien Querlioz (Univ Paris-Sud) &middot; Julie Grollier (Unité Mixte CNRS/Thalès)</i></p>

      <p><b>Universal Invariant and Equivariant Graph Neural Networks</b><br><i>Nicolas Keriven (Ecole Normale Supérieure) &middot; Gabriel Peyré (CNRS and ENS)</i></p>

      <p><b>The bias of the sample mean in multi-armed bandits can be positive or negative</b><br><i>Jaehyeok Shin (Carnegie Mellon University) &middot; Aaditya Ramdas (Carnegie Mellon University) &middot; Alessandro Rinaldo (CMU)</i></p>

      <p><b>On the Correctness and Sample Complexity of Inverse Reinforcement Learning</b><br><i>Abi Komanduru (Purdue University) &middot; Jean Honorio (Purdue University)</i></p>

      <p><b>VIREL: A Variational Inference Framework for Reinforcement Learning</b><br><i>Matthew Fellows (University of Oxford) &middot; Anuj Mahajan (University of Oxford) &middot; Tim G. J. Rudner (University of Oxford) &middot; Shimon Whiteson (University of Oxford)</i></p>

      <p><b>First Order Motion Model for Image Animation</b><br><i>Aliaksandr Siarohin (University of Trento) &middot; Stephane Lathuillere (University of Trento) &middot; Sergey Tulyakov (Snap Inc) &middot; Elisa Ricci (FBK - Technologies of Vision) &middot; Nicu Sebe (University of Trento)</i></p>

      <p><b>Tensor Monte Carlo: Particle Methods for the GPU era</b><br><i>Laurence Aitchison (University of Cambridge)</i></p>

      <p><b>Unsupervised Emergence of Egocentric Spatial Structure from Sensorimotor Prediction</b><br><i>Alban Laflaquière (ISIR) &middot; Michael Garcia Ortiz (SoftBank Robotics Europe)</i></p>

      <p><b>Learning from Label Proportions with Generative Adversarial Networks</b><br><i>Jiabin Liu (University of Chinese Academy of Sciences) &middot; Bo Wang (University of International Business and Economics) &middot; Zhiquan Qi (University of Chinese Academy of Sciences) &middot; YingJie Tian (University of Chinese Academy of Sciences) &middot; Yong Shi (University of Chinese Academy of Sciences)</i></p>

      <p><b>Efficient and Thrifty Voting by Any Means Necessary</b><br><i>Debmalya Mandal (Columbia University) &middot; Ariel D Procaccia (Carnegie Mellon University) &middot; Nisarg Shah (University of Toronto) &middot; David Woodruff (Carnegie Mellon University)</i></p>

      <p><b>PointDAN: A Multi-Scale 3D Domain Adaption Network for Point Cloud Representation</b><br><i>Can Qin (Northeastern University) &middot; Haoxuan You (Columbia University) &middot; Lichen Wang (Northeastern University) &middot; C.-C. Jay Kuo (University of Southern California) &middot; Yun Fu (Northeastern University)</i></p>

      <p><b>ZO-AdaMM: Zeroth-Order Adaptive Momentum Method for Black-Box Optimization</b><br><i>Xiangyi Chen (University of Minnesota) &middot; Sijia Liu (MIT-IBM Watson AI Lab, IBM Research AI) &middot; Kaidi Xu (Northeastern University) &middot; Xingguo Li (Princeton University) &middot; Xue Lin (Northeastern University) &middot; Mingyi Hong (University of Minnesota) &middot; David Cox (MIT-IBM Watson AI Lab)</i></p>

      <p><b>Non-Stationary Markov Decision Processes, a Worst-Case Approach using Model-Based Reinforcement Learning</b><br><i>Erwan Lecarpentier (Université de Toulouse, ONERA The French Aerospace Lab) &middot; Emmanuel Rachelson (ISAE-SUPAERO / University of Toulouse)</i></p>

      <p><b>Depth-First Proof-Number Search with Heuristic Edge Cost and Application to Chemical Synthesis Planning</b><br><i>Akihiro Kishimoto (IBM Research) &middot; Beat Buesser (IBM Research) &middot; Bei Chen (IBM Research) &middot; Adi Botea (IBM Research)</i></p>

      <p><b>Toward a Characterization of Loss Functions for Distribution Learning</b><br><i>Nika Haghtalab (Microsoft) &middot; Cameron Musco (Microsoft Research) &middot; Bo Waggoner (U. Colorado, Boulder)</i></p>

      <p><b>Coresets for Archetypal Analysis</b><br><i>Sebastian Mair (Leuphana University) &middot; Ulf Brefeld (Leuphana)</i></p>

      <p><b>Emergence of Object Segmentation in Perturbed Generative Models</b><br><i>Adam Bielski (University of Bern) &middot; Paolo Favaro (Bern University, Switzerland)</i></p>

      <p><b>Optimal Sparse Decision Trees</b><br><i>Xiyang Hu (Duke University) &middot; Cynthia Rudin (Duke) &middot; Margo Seltzer (University of British Columbia)</i></p>

      <p><b>Escaping from saddle points on Riemannian manifolds</b><br><i>Yue Sun (University of Washington) &middot; Nicolas Flammarion (UC Berkeley) &middot; Maryam Fazel (University of Washington)</i></p>

      <p><b>Muti-source Domain Adaptation for Semantic Segmentation</b><br><i>Sicheng Zhao (University of California Berkeley) &middot; Bo Li (Harbin Institute of Technology) &middot; Xiangyu Yue (UC Berkeley) &middot; Yang Gu (Didi chuxing) &middot; Pengfei Xu (Didi Chuxing) &middot; Runbo Hu (DiDi Chuxing) &middot; Hua Chai (Didi Chuxing) &middot; Kurt Keutzer (EECS, UC Berkeley)</i></p>

      <p><b>Localized Structured Prediction</b><br><i>Carlo Ciliberto (Imperial College London) &middot; Francis Bach (INRIA - Ecole Normale Superieure) &middot; Alessandro Rudi (INRIA, Ecole Normale Superieure)</i></p>

      <p><b>Nonzero-sum Adversarial Hypothesis Testing Games</b><br><i>Sarath Yasodharan (Indian Institute of Science) &middot; Patrick Loiseau (Inria)</i></p>

      <p><b>Manifold-regression to predict from MEG/EEG brain signals without source modeling</b><br><i>David Sabbagh (INRIA) &middot; Pierre Ablin (Inria) &middot; Gael Varoquaux (Parietal Team, INRIA) &middot; Alexandre Gramfort (INRIA, Université Paris-Saclay) &middot; Denis A. Engemann (INRIA Saclay)</i></p>

      <p><b>Modeling Tabular data using Conditional GAN</b><br><i>Lei Xu (MIT) &middot; Maria Skoularidou (University of Cambridge) &middot; Alfredo Cuesta Infante (Universidad Rey Juan Carlos) &middot; Kalyan Veeramachaneni (Massachusetts Institute of Technology)</i></p>

      <p><b>Normalization Helps Training of Quantized LSTM</b><br><i>Lu Hou (Huawei Technologies Co., Ltd) &middot; Jinhua Zhu (University of Science and Technology of China) &middot; James Kwok (Hong Kong University of Science and Technology) &middot; Fei Gao (University of Chinese Academy of Sciences) &middot; Tao Qin (Microsoft Research) &middot; Tie-Yan Liu (Microsoft Research)</i></p>

      <p><b>Trajectory of Alternating Direction Method of Multipliers and Adaptive Acceleration</b><br><i>Clarice Poon (University of Bath) &middot; Jingwei Liang (DAMTP, University of Cambridge)</i></p>

      <p><b>Deep Scale-spaces: Equivariance Over Scale</b><br><i>Daniel Worrall (University of Amsterdam) &middot; Max Welling (University of Amsterdam / Qualcomm AI Research)</i></p>

      <p><b>GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series</b><br><i>Edward De Brouwer (KU Leuven) &middot; Jaak Simm (KU Leuven) &middot; Adam Arany (University of Leuven) &middot; Yves Moreau (KU Leuven)</i></p>

      <p><b>Estimating Convergence of Markov chains with L-Lag Couplings</b><br><i>Niloy Biswas (Harvard University) &middot; Pierre E Jacob (Harvard University)</i></p>

      <p><b>Learning-Based Low-Rank Approximations</b><br><i>Piotr Indyk (MIT) &middot; Ali Vakilian (Massachusetts Institute of Technology) &middot; Yang Yuan (Cornell University)</i></p>

      <p><b>Implicit Regularization in Deep Matrix Factorization</b><br><i>Sanjeev Arora (Princeton University) &middot; Nadav Cohen (Tel Aviv University) &middot; Wei Hu (Princeton University) &middot; Yuping Luo (Princeton University)</i></p>

      <p><b>List-decodable Linear Regression</b><br><i>Sushrut Karmalkar (The University of Texas at Austin) &middot; Adam Klivans (UT Austin) &middot; Pravesh Kothari (Princeton University and Institute for Advanced Study)</i></p>

      <p><b>Learning elementary structures for 3D shape generation and matching</b><br><i>Theo Deprelle (École des ponts ParisTech) &middot; Thibault Groueix (École des ponts ParisTech) &middot; Matthew Fisher (Adobe Research) &middot; Vladimir Kim (Adobe) &middot; Bryan Russell (Adobe) &middot; Mathieu Aubry (École des ponts ParisTech)</i></p>

      <p><b>On the Hardness of Robust Classification</b><br><i>Pascale Gourdeau (University of Oxford) &middot; Varun Kanade (University of Oxford) &middot; Marta Kwiatkowska (University of Oxford) &middot; James Worrell (University of Oxford)</i></p>

      <p><b>Foundations of Comparison-Based Hierarchical Clustering</b><br><i>Debarghya Ghoshdastidar (University of Tübingen) &middot; Michaël Perrot (Max Planck Institute for Intelligent Systems) &middot; Ulrike von Luxburg (University of Tübingen)</i></p>

      <p><b>What the Vec? Towards Probabilistically Grounded Embeddings</b><br><i>Carl Allen (University of Edinburgh) &middot; Ivana Balazevic (University of Edinburgh) &middot; Timothy Hospedales (University of Edinburgh)</i></p>

      <p><b>Minimizers of the Empirical Risk and Risk Monotonicity</b><br><i>Marco Loog (Delft University of Technology) &middot; Tom Viering (Delft University of Technology, Netherlands) &middot; Alexander Mey (TU Delft)</i></p>

      <p><b>Explicit Planning for Efficient Exploration in Reinforcement Learning</b><br><i>Liangpeng Zhang (University of Birmingham) &middot; Xin Yao (University of Birmingham)</i></p>

      <p><b>Lower Bounds on Adversarial Robustness from Optimal Transport</b><br><i>Arjun Nitin Bhagoji (Princeton University) &middot; Daniel Cullina (Princeton University) &middot; Prateek Mittal (Princeton University)</i></p>

      <p><b>Neural Spline Flows</b><br><i>Conor Durkan (University of Edinburgh) &middot; Arturs Bekasovs (University of Edinburgh) &middot; Iain Murray (University of Edinburgh) &middot; George Papamakarios (DeepMind)</i></p>

      <p><b>Phase Transitions and Cyclic Phenomena in Bandits with Switching Constraints</b><br><i>David Simchi-Levi (MIT) &middot; Yunzong Xu (MIT)</i></p>

      <p><b>Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization</b><br><i>Koen Helwegen (Plumerai) &middot; James Widdicombe (Plumerai) &middot; Lukas Geiger (Plumerai) &middot; Zechun Liu (HKUST) &middot; Kwang-Ting Cheng (Hong Kong University of Science and Technology) &middot; Koen Helwegen (Plumerai)</i></p>

      <p><b>Nonlinear scaling of resource allocation in sensory bottlenecks</b><br><i>Laura R Edmondson (University of Sheffield) &middot; Alejandro Jimenez Rodriguez (University of Sheffield) &middot; Hannes P. Saal (University of Sheffield)</i></p>

      <p><b>Constrained Reinforcement Learning: A Dual Approach</b><br><i>Santiago Paternain (University of Pennsylvania) &middot; Luiz Chamon (University of Pennsylvania) &middot; Miguel Calvo-Fullana (University of Pennsylvania) &middot; Alejandro Ribeiro (University of Pennsylvania)</i></p>

      <p><b>Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules</b><br><i>Niklas Gebauer (Technische Universität Berlin) &middot; Michael Gastegger (Technische Universität Berlin) &middot; Kristof Schütt (TU Berlin)</i></p>

      <p><b>An adaptive nearest neighbor rule for classification</b><br><i>Akshay Balsubramani (Stanford) &middot; Sanjoy Dasgupta (UC San Diego) &middot; yoav S Freund (UCSD) &middot; Shay Moran (IAS, Princeton)</i></p>

      <p><b>Coresets for Clustering with Fairness Constraints</b><br><i>Lingxiao Huang (EPFL) &middot; Shaofeng H.-C. Jiang (Weizmann Institute of Science) &middot; Nisheeth Vishnoi (Yale University)</i></p>

      <p><b>PerspectiveNet: A Scene-consistent Image Generator for New View Synthesis in Real Indoor Environments</b><br><i>Ben Graham (Facebook Research) &middot; David Novotny (Facebook AI Research) &middot; Jeremy Reizenstein (Facebook AI Research)</i></p>

      <p><b>MAVEN: Multi-Agent Variational Exploration</b><br><i>Anuj Mahajan (University of Oxford) &middot; Tabish Rashid (University of Oxford) &middot; Mikayel Samvelyan (Russian-Armenian University) &middot; Shimon Whiteson (University of Oxford)</i></p>

      <p><b>Competitive Gradient Descent</b><br><i>Florian Schaefer (Caltech) &middot; Anima Anandkumar (NVIDIA / Caltech)</i></p>

      <p><b>Globally Convergent Newton Methods for Ill-conditioned Generalized Self-concordant Losses</b><br><i>Ulysse Marteau-Ferey (INRIA) &middot; Francis Bach (INRIA - Ecole Normale Superieure) &middot; Alessandro Rudi (INRIA, Ecole Normale Superieure)</i></p>

      <p><b>Continual Unsupervised Representation Learning</b><br><i>Dushyant Rao (DeepMind) &middot; Francesco Visin (DeepMind) &middot; Andrei Rusu (DeepMind) &middot; Razvan Pascanu (Google DeepMind) &middot; Yee Whye Teh (University of Oxford, DeepMind) &middot; Raia Hadsell (DeepMind)</i></p>

      <p><b>Self-Routing Capsule Networks</b><br><i>Taeyoung Hahn (SNUVL) &middot; Myeongjang Pyeon (Seoul National University) &middot; Gunhee Kim (Seoul National University)</i></p>

      <p><b>The Parameterized Complexity of Cascading Portfolio Scheduling</b><br><i>Eduard Eiben (University of Bergen) &middot; Robert Ganian (TU Wien) &middot; Iyad Kanj (DePaul University, Chicago) &middot; Stefan Szeider (Vienna University of Technology)</i></p>

      <p><b>Maximum Expected Hitting Cost of a Markov Decision Process and Informativeness of Rewards</b><br><i>Zhongtian Dai (Toyota Technological Institute at Chicago) &middot; Matthew R. Walter (TTI-Chicago)</i></p>

      <p><b>Bipartite expander Hopfield networks as self-decoding high-capacity error correcting codes</b><br><i>Rishidev Chaudhuri (University of California, Davis) &middot; Ila Fiete (University of Texas at Austin)</i></p>

      <p><b>Sequence Modelling with Unconstrained Generation Order</b><br><i>Dmitriy Emelyanenko (Yandex; National Research University Higher School of Economics) &middot; Elena Voita (Yandex; University of Amsterdam) &middot; Pavel Serdyukov (Yandex)</i></p>

      <p><b>Probabilistic Logic Neural Networks for Reasoning</b><br><i>Meng Qu (MILA) &middot; Jian Tang (HEC Montreal & MILA)</i></p>

      <p><b>A Polynomial Time Algorithm for Log-Concave Maximum Likelihood via Locally Exponential Families</b><br><i>Brian Axelrod (Stanford) &middot; Ilias Diakonikolas (USC) &middot; Alistair Stewart (University of Southern California) &middot; Anastasios Sidiropoulos (University of Illinois at Chicago) &middot; Gregory Valiant (Stanford University)</i></p>

      <p><b>A Unifying Framework for Spectrum-Preserving Graph Sparsification and Coarsening</b><br><i>Gecia Bravo Hermsdorff (Princeton University) &middot; Lee Gunderson (Princeton University)</i></p>

      <p><b>Stochastic Runge-Kutta Accelerates Langevin Monte Carlo and Beyond</b><br><i>Xuechen Li (Google) &middot; Yi Wu (University of Toronto & Vector Institute) &middot; Lester Mackey (Microsoft Research) &middot; Murat Erdogdu (University of Toronto)</i></p>

      <p><b>The Implicit Bias of AdaGrad on Separable Data</b><br><i>Qian Qian (the Ohio State University) &middot; Xiaoyuan Qian (Dalian University of Technology)</i></p>

      <p><b>On two ways to use determinantal point processes for Monte Carlo integration</b><br><i>Guillaume Gautier (CNRS, INRIA, Univ. Lille) &middot; Rémi Bardenet (University of Lille) &middot; Michal Valko (DeepMind Paris and Inria Lille - Nord Europe)</i></p>

      <p><b>LiteEval: A Coarse-to-Fine Framework for Resource Efficient Video Recognition</b><br><i>Zuxuan Wu (UMD) &middot; Caiming Xiong (Salesforce) &middot; Yu-Gang Jiang (Fudan University) &middot; Larry Davis (University of Maryland)</i></p>

      <p><b>How degenerate is the parametrization of neural networks with the ReLU activation function?</b><br><i>Dennis Elbrächter (University of Vienna) &middot; Julius Berner (University of Vienna) &middot; Philipp Grohs (University of Vienna)</i></p>

      <p><b>Spike-Train Level Backpropagation for Training Deep Recurrent Spiking Neural Networks</b><br><i>Wenrui Zhang (Texas A&M University) &middot; Peng Li (Texas A&M University)</i></p>

      <p><b>Re-examination of the Role of Latent Variables in Sequence Modeling</b><br><i>Guokun Lai (Carnegie Mellon University) &middot; Zihang Dai (Carnegie Mellon University)</i></p>

      <p><b>Max-value Entropy Search for Multi-Objective Bayesian Optimization</b><br><i>Syrine Belakaria (Washington State University) &middot; Aryan Deshwal (Washington State University) &middot; Janardhan Rao Doppa (Washington State University)</i></p>

      <p><b>Stein Variational Gradient Descent With Matrix-Valued Kernels</b><br><i>Dilin Wang (UT Austin) &middot; Ziyang Tang (UT Austin) &middot; Chandrajit Bajaj (The University of Texas at Austin) &middot; Qiang Liu (UT Austin)</i></p>

      <p><b>Crowdsourcing via Pairwise Co-occurrences: Identifiability and Algorithms</b><br><i>Shahana Ibrahim (Oregon State University) &middot; Xiao Fu (Oregon State University) &middot; Nikolaos Kargas (University of Minnesota) &middot; Kejun Huang (University of Florida)</i></p>

      <p><b>Detecting Overfitting via Adversarial Examples</b><br><i>Roman Werpachowski (DeepMind) &middot; András György (DeepMind) &middot; Csaba Szepesvari (DeepMind/University of Alberta)</i></p>

      <p><b>A Unified Bellman Optimality Principle Combining Reward Maximization and Empowerment</b><br><i>Felix Leibfried (PROWLER.io) &middot; Sergio Pascual-Diaz (PROWLER.io) &middot; Jordi Grau-Moya (PROWLER.io)</i></p>

      <p><b>SMILe: Scalable Meta Inverse Reinforcement Learning through Context-Conditional Policies</b><br><i>Seyed Kamyar Seyed Ghasemipour (University of Toronto) &middot; Shixiang (Shane) Gu (Google Brain) &middot; Richard Zemel (Vector Institute/University of Toronto)</i></p>

      <p><b>Towards Understanding the Importance of Shortcut Connections in Residual Networks</b><br><i>Tianyi Liu (Georgia Institute of Technolodgy) &middot; Minshuo Chen (Georgia Tech) &middot; Mo Zhou (Duke University) &middot; Simon Du (Carnegie Mellon University) &middot; Enlu Zhou (Georgia Institute of Technology) &middot; Tuo Zhao (Gatech)</i></p>

      <p><b>Modular Universal Reparameterization: Deep Multi-task Learning Across Diverse Domains</b><br><i>Elliot Meyerson (Cognizant) &middot; Risto Miikkulainen (The University of Texas at Austin; Cognizant)</i></p>

      <p><b>Solving Interpretable Kernel Dimensionality Reduction</b><br><i>Chieh T Wu (Northeastern University) &middot; Jared Miller (Northeastern University) &middot; Yale Chang (Northeastern University) &middot; Mario Sznaier (Northeastern University) &middot; Jennifer Dy (Northeastern University)</i></p>

      <p><b>Interaction Hard Thresholding: Consistent Sparse Quadratic Regression in Sub-quadratic Time and Space</b><br><i>Shuo Yang (UT Austin) &middot; Yanyao Shen (UT Austin) &middot; Sujay Sanghavi (UT-Austin)</i></p>

      <p><b>A Model to Search for Synthesizable Molecules</b><br><i>John Bradshaw (University of Cambridge/MPI Tuebingen) &middot; Brooks Paige (Alan Turing Institute) &middot; Matt J Kusner (University College London) &middot; Marwin Segler (BenevolentAI) &middot; José Miguel Hernández-Lobato (University of Cambridge)</i></p>

      <p><b>Post training 4-bit quantization of convolutional networks for rapid-deployment</b><br><i>Ron Banner (Intel - Artificial Intelligence Products Group (AIPG)) &middot; Yury Nahshan (Intel corp.) &middot; Daniel Soudry (Technion)</i></p>

      <p><b>Fast and Flexible Multi-Task Classification using Conditional Neural Adaptive Processes</b><br><i>James Requeima (University of Cambridge / Invenia Labs) &middot; Jonathan Gordon (University of Cambridge) &middot; John Bronskill (University of Cambridge) &middot; Sebastian Nowozin (Microsoft Research) &middot; Richard Turner (Cambridge)</i></p>

      <p><b>Differentially Private Anonymized Histograms</b><br><i>Ananda Theertha Suresh (Google)</i></p>

      <p><b>Dynamic Local Regret for Non-convex Online Forecasting</b><br><i>Sergul Aydore (Stevens Institute of Technology) &middot; Tianhao Zhu (Stevens Institute of Techonlogy) &middot; Dean Foster (Amazon)</i></p>

      <p><b>Learning Local Search Heuristics for Boolean Satisfiability</b><br><i>Emre Yolcu (Carnegie Mellon University) &middot; Barnabas Poczos (Carnegie Mellon University)</i></p>

      <p><b>Provably Efficient Q-Learning with Low Switching Cost</b><br><i>Yu Bai (Stanford University) &middot; Tengyang Xie (University of Illinois at Urbana-Champaign) &middot; Nan Jiang (University of Illinois at Urbana-Champaign) &middot; Yu-Xiang Wang (UC Santa Barbara)</i></p>

      <p><b>Solving graph compression via optimal transport</b><br><i>Vikas Garg (MIT) &middot; Tommi Jaakkola (MIT)</i></p>

      <p><b>PyTorch: An Imperative Style, High-Performance Deep Learning Library</b><br><i>Benoit Steiner (Facebook AI Research) &middot; Zachary DeVito (Facebook AI Research) &middot; Soumith Chintala (Facebook AI Research) &middot; Sam Gross (Facebook) &middot; Adam Paszke (University of Warsaw) &middot; Francisco Massa (Facebook AI Research) &middot; Adam Lerer (Facebook AI Research) &middot; Gregory Chanan (Facebook) &middot; Zeming Lin (Facebook AI Research) &middot; Edward Yang (Facebook) &middot; Alban Desmaison (Oxford University) &middot; Alykhan Tejani (Twitter, Inc.) &middot; Andreas Kopf (Xamla) &middot; James Bradbury (Google Brain) &middot; Luca Antiga (Orobix) &middot; Martin Raison (Nabla) &middot; Natalia Gimelshein (NVIDIA) &middot; Sasank Chilamkurthy (Qure.ai) &middot; Trevor Killeen (Self Employed) &middot; Lu Fang (Facebook) &middot; Junjie Bai (Facebook)</i></p>

      <p><b>Stability of Graph Scattering Transforms</b><br><i>Fernando Gama (University of Pennsylvania) &middot; Alejandro Ribeiro (University of Pennsylvania) &middot; Joan Bruna (NYU)</i></p>

      <p><b>A Debiased MDI Feature Importance Measure for Random Forests</b><br><i>Xiao Li (University of California, Berkeley) &middot; Yu Wang (UC Berkeley) &middot; Sumanta Basu (Cornell University) &middot; Karl Kumbier (University of California, Berkeley) &middot; Bin Yu (UC Berkeley)</i></p>

      <p><b>Difference Maximization Q-learning: Provably Efficient Q-learning with Function Approximation</b><br><i>Simon Du (Carnegie Mellon University) &middot; Yuping Luo (Princeton University) &middot; Ruosong Wang (Carnegie Mellon University) &middot; Hanrui Zhang (Duke University)</i></p>

      <p><b>Sparse Logistic Regression Learns All Discrete Pairwise Graphical Models</b><br><i>Shanshan Wu (University of Texas at Austin) &middot; Sujay Sanghavi (UT-Austin) &middot; Alexandros Dimakis (University of Texas, Austin)</i></p>

      <p><b>Fast Convergence of Natural Gradient Descent for Over-Parameterized Neural Networks</b><br><i>Guodong Zhang (University of Toronto) &middot; James Martens (DeepMind) &middot; Roger Grosse (University of Toronto)</i></p>

      <p><b>Rapid Convergence of the Unadjusted Langevin Algorithm: Log-Sobolev Suffices</b><br><i>Santosh Vempala (Georgia Tech) &middot; Andre Wibisono ()</i></p>

      <p><b>Learning Distributions Generated by One-Layer ReLU Networks</b><br><i>Shanshan Wu (University of Texas at Austin) &middot; Alexandros Dimakis (University of Texas, Austin) &middot; Sujay Sanghavi (UT-Austin)</i></p>

      <p><b>Large-scale optimal transport map estimation using projection pursuit</b><br><i>Cheng Meng (University of Georgia) &middot; Yuan Ke (University of Georgia) &middot; Jingyi Zhang (The University of Georgia) &middot; Mengrui Zhang (University of Georgia) &middot; Wenxuan Zhong () &middot; Ping Ma (University of Georgia)</i></p>

      <p><b>A Structured Prediction Approach for Generalization in Cooperative Multi-Agent Reinforcement Learning</b><br><i>Nicolas Carion (Facebook AI Research Paris) &middot; Nicolas Usunier (Facebook AI Research) &middot; Gabriel Synnaeve (Facebook) &middot; Alessandro Lazaric (Facebook Artificial Intelligence Research)</i></p>

      <p><b>On Exact Computation with an Infinitely Wide Neural Net</b><br><i>Sanjeev Arora (Princeton University) &middot; Simon Du (Carnegie Mellon University) &middot; Wei Hu (Princeton University) &middot; zhiyuan li (Princeton University) &middot; Ruslan Salakhutdinov (Carnegie Mellon University) &middot; Ruosong Wang (Carnegie Mellon University)</i></p>

      <p><b>Loaded DiCE: Trading off Bias and Variance in Any-Order Score Function Gradient Estimators for Reinforcement Learning</b><br><i>Gregory Farquhar (University of Oxford) &middot; Shimon Whiteson (University of Oxford) &middot; Jakob Foerster (University of Oxford)</i></p>

      <p><b>Chirality Nets for Human Pose Regression</b><br><i>Raymond Yeh (University of Illinois at Urbana–Champaign) &middot; Yuan-Ting Hu (University of Illinois Urbana-Champaign) &middot; Alexander Schwing (University of Illinois at Urbana-Champaign)</i></p>

      <p><b>Efficient Approximation of Deep ReLU Networks for Functions on Low Dimensional Manifolds</b><br><i>Minshuo Chen (Georgia Tech) &middot; Haoming Jiang (Georgia Institute of Technology) &middot; Wenjing Liao (Georgia Tech) &middot; Tuo Zhao (Georgia Tech)</i></p>

      <p><b>Fast Decomposable Submodular Function Minimization using Constrained Total Variation</b><br><i>Senanayak Sesh Kumar Karri (Imperial College London) &middot; Francis Bach (INRIA - Ecole Normale Superieure) &middot; Thomas Pock (Graz University of Technology)</i></p>

      <p><b>Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model</b><br><i>Guodong Zhang (University of Toronto) &middot; Lala Li (Google) &middot; Zachary Nado (Google Inc.) &middot; James Martens (DeepMind) &middot; Sushant Sachdeva (University of Toronto) &middot; George Dahl (Google Brain) &middot; Chris Shallue (Google Brain) &middot; Roger Grosse (University of Toronto)</i></p>

      <p><b>Spherical Text Embedding</b><br><i>Yu Meng (University of Illinois at Urbana-Champaign) &middot; Jiaxin Huang (University of Illinois Urbana-Champaign) &middot; Guangyuan Wang (UIUC) &middot; Chao Zhang (Georgia Institute of Technology) &middot; Honglei Zhuang (Google Research) &middot; Lance Kaplan (U.S. Army Research Laboratory) &middot; Jiawei Han (UIUC)</i></p>

      <p><b>Möbius Transformation for Fast Inner Product Search on Graph</b><br><i>Zhixin Zhou (Baidu Research) &middot; Shulong Tan (Baidu Research) &middot; Zhaozhuo Xu (Baidu Research) &middot; Ping Li (Baidu Research USA)</i></p>

      <p><b>Hyperbolic Graph Neural Networks</b><br><i>Qi Liu (National University of Singapore) &middot; Maximilian Nickel (Facebook AI Research) &middot; Douwe Kiela (Facebook AI Research)</i></p>

      <p><b>Average Individual Fairness: Algorithms, Generalization and Experiments</b><br><i>Saeed Sharifi-Malvajerdi (University of Pennsylvania) &middot; Michael Kearns (University of Pennsylvania) &middot; Aaron Roth (University of Pennsylvania)</i></p>

      <p><b>Fixing the train-test resolution discrepancy</b><br><i>Hugo Touvron (Facebook AI Research) &middot; Andrea Vedaldi (Facebook AI Research and University of Oxford) &middot; Matthijs Douze (Facebook AI Research) &middot; Herve Jegou (Facebook AI Research)</i></p>

      <p><b>Modeling Dynamic Functional Connectivity with Latent Factor Gaussian Processes</b><br><i>Lingge Li (UC Irvine) &middot; Dustin Pluta (UC Irvine) &middot; Babak Shahbaba (UCI) &middot; Norbert Fortin (UC Irvine) &middot; Hernando Ombao (KAUST) &middot; Pierre Baldi (UC Irvine)</i></p>

      <p><b>Manipulating a Learning Defender and Ways to Counteract</b><br><i>Jiarui Gan (University of Oxford) &middot; Qingyu Guo (Nanyang Technological University) &middot; Long Tran-Thanh (University of Southampton) &middot; Bo An (Nanyang Technological University) &middot; Michael Wooldridge (Univ of Oxford)</i></p>

      <p><b>Learning-In-The-Loop Optimization: End-To-End Control And Co-Design Of Soft Robots Through Learned Deep Latent Representations</b><br><i>Andrew Spielberg (Massachusetts Institute of Technology) &middot; Allan Zhao (Massachusetts Institute of Technology) &middot; Yuanming Hu (Massachusetts Institute of Technology) &middot; Tao Du (MIT) &middot; Wojciech Matusik (MIT) &middot; Daniela Rus (Massachusetts Institute of Technology)</i></p>

      <p><b>Learning to Infer Implicit Surfaces without 3D Supervision</b><br><i>Shichen Liu (Tsinghua University) &middot; Shunsuke Saito (University of Southern California) &middot; Weikai Chen (USC Institute for Creative Technology) &middot; Hao Li (Pinscreen/University of Southern California/USC ICT)</i></p>

      <p><b>Fast and Accurate Least-Mean-Squares Solvers</b><br><i>Ibrahim Jubran (The University of Haifa) &middot; Alaa Maalouf (The University of Haifa) &middot; Dan Feldman (University of Haifa)</i></p>

      <p><b>Certifiable Robustness to Graph Perturbations</b><br><i>Aleksandar Bojchevski (Technical University of Munich) &middot; Stephan Günnemann (Technical University of Munich)</i></p>

      <p><b>Fast Convergence of Belief Propagation to Global Optima: Beyond Correlation Decay </b><br><i>Frederic Koehler (MIT)</i></p>

      <p><b>Paradoxes in Fair Machine Learning</b><br><i>Paul Goelz (Carnegie Mellon University) &middot; Anson Kahng (Carnegie Mellon University) &middot; Ariel D Procaccia (Carnegie Mellon University)</i></p>

      <p><b>Provably Global Convergence of Actor-Critic: A Case for Linear Quadratic Regulator with Ergodic Cost</b><br><i>Zhuoran Yang (Princeton University) &middot; Yongxin Chen (Georgia Institute of Technology) &middot; Mingyi Hong (University of Minnesota) &middot; Zhaoran Wang (Northwestern University)</i></p>

      <p><b>The spiked matrix model with generative priors</b><br><i>Benjamin Aubin (Ipht Saclay) &middot; Bruno Loureiro (IPhT Saclay) &middot; Antoine Maillard (Ecole Normale Supérieure) &middot; Florent Krzakala (ENS Paris & Sorbonnes Université) &middot; Lenka Zdeborová (CEA Saclay)</i></p>

      <p><b>Gradient Dynamics of Shallow Low-Dimensional ReLU Networks</b><br><i>Francis Williams (New York University) &middot; Matthew Trager (NYU) &middot; Daniele Panozzo (NYU) &middot; Claudio Silva (New York University) &middot; Denis Zorin (New York University) &middot; Joan Bruna (NYU)</i></p>

      <p><b>Robust and Communication-Efficient Collaborative Learning</b><br><i>Amirhossein Reisizadeh (UC Santa Barbara) &middot; Hossein Taheri (UCSB) &middot; Aryan Mokhtari (UT Austin) &middot; Hamed Hassani (UPenn) &middot; Ramtin Pedarsani (UC Santa Barbara)</i></p>

      <p><b>Multiclass Learning from Contradictions</b><br><i>Sauptik Dhar (LG Electronics) &middot; Vladimir Cherkassky (University of Minnesota) &middot; Mohak Shah (LG Electronics)</i></p>

      <p><b>Learning from Trajectories via Subgoal Discovery</b><br><i>Sujoy Paul (UC Riverside) &middot; Jeroen Vanbaar (MERL (Mitsubishi Electric Research Laboratories), Cambridge MA) &middot; Amit Roy-Chowdhury  (University of California, Riverside, USA )</i></p>

      <p><b>Distributed Low-rank Matrix Factorization With Exact Consensus</b><br><i>Zhihui Zhu (Johns Hopkins University) &middot; Qiuwei Li (Colorado School of Mines) &middot; Xinshuo Yang (Colorado School of Mines) &middot; Gongguo Tang (Colorado School of Mines) &middot; Michael B Wakin (Colorado School of Mines)</i></p>

      <p><b>Online Normalization for Training Neural Networks</b><br><i>Vitaliy  Chiley (Cerebras Systems) &middot; Ilya Sharapov (Cerebras Systems) &middot; Atli Kosson (Cerebras Systems) &middot; Urs Koster (Cerebras Systems) &middot; Ryan Reece (Cerebras Systems) &middot; Sofia Samaniego de la Fuente (Cerebras Systems) &middot; Vishal Subbiah (Cerebras Systems) &middot; Michael James (Cerebras)</i></p>

      <p><b>The Synthesis of XNOR Recurrent Neural Networks with Stochastic Logic</b><br><i>Arash Ardakani (McGill University) &middot; Zhengyun Ji (McGill University) &middot; Amir Ardakani (McGill University) &middot; Warren Gross (McGill University)</i></p>

      <p><b>An adaptive Mirror-Prox method for variational inequalities with singular operators</b><br><i>Kimon Antonakopoulos (Inria) &middot; Veronica Belmega (ENSEA) &middot; Panayotis Mertikopoulos (CNRS (French National Center for Scientific Research))</i></p>

      <p><b>N-Gram Graph: A Simple Unsupervised Representation for Molecules</b><br><i>Shengchao Liu (UW-Madison) &middot; Mehmet F Demirel (University of Wisconsin-Madison) &middot; Yingyu Liang (University of Wisconsin Madison)</i></p>

      <p><b>Characterizing the exact behaviors of temporal difference learning algorithms using Markov jump linear system theory</b><br><i>Bin Hu (University of Illinois at Urbana-Champaign) &middot; Usman A Syed (University of Illinois Urbana Champaign)</i></p>

      <p><b>Facility Location Problem in Differential Privacy Model Revisited </b><br><i>Yunus Esencayi (State University of New York at Buffalo) &middot; Marco Gaboardi (Univeristy at Buffalo) &middot; Shi Li (University at Buffalo) &middot; Di Wang (State University of New York at Buffalo)</i></p>

      <p><b>Revisiting Auxiliary Latent Variables in Generative Models</b><br><i>John Lawson (New York University) &middot; George Tucker (Google Brain) &middot; Bo Dai (Google Brain) &middot; Rajesh Ranganath (New York University)</i></p>

      <p><b>Finite-time Analysis of Approximate Policy Iteration for the Linear Quadratic Regulator</b><br><i>Karl Krauth (UC berkeley) &middot; Stephen Tu (UC Berkeley) &middot; Benjamin Recht (UC Berkeley)</i></p>

      <p><b>A Universally Optimal Multistage Accelerated Stochastic Gradient Method</b><br><i>Necdet Serhat Aybat (Penn State University) &middot; Alireza Fallah (MIT) &middot; Mert Gurbuzbalaban (Rutgers) &middot; Asuman Ozdaglar (Massachusetts Institute of Technology)</i></p>

      <p><b>From deep learning to mechanistic understanding in neuroscience: the structure of retinal prediction</b><br><i>Hidenori Tanaka (Stanford) &middot; Aran Nayebi (Stanford University) &middot; Stephen Baccus (Stanford University) &middot; Surya Ganguli (Stanford)</i></p>

      <p><b>Large Memory Layers with Product Keys</b><br><i>Guillaume Lample (Facebook AI Research) &middot; Alexandre Sablayrolles (Facebook AI Research) &middot; Marc'Aurelio Ranzato (Facebook AI Research) &middot; Ludovic Denoyer (Facebook - FAIR) &middot; Herve Jegou (Facebook AI Research)</i></p>

      <p><b>Learning Deterministic Weighted Automata with Queries and Counterexamples</b><br><i>Gail Weiss (Technion) &middot; Yoav Goldberg (Bar Ilan University) &middot; Eran Yahav (Technion)</i></p>

      <p><b>Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent</b><br><i>Jaehoon Lee (Google Brain) &middot; Lechao Xiao (Google Brain) &middot; Samuel Schoenholz (Google Brain) &middot; Yasaman Bahri (Google Brain) &middot; Roman Novak (Google Brain) &middot; Jascha Sohl-Dickstein (Google Brain) &middot; Jeffrey Pennington (Google Brain)</i></p>

      <p><b>Time/Accuracy Tradeoffs for Learning a ReLU with respect to Gaussian Marginals</b><br><i>Surbhi Goel (UT Austin) &middot; Sushrut Karmalkar (The University of Texas at Austin) &middot; Adam Klivans (UT Austin)</i></p>

      <p><b>Visualizing and Measuring the Geometry of BERT</b><br><i>Emily Reif (Google) &middot; Ann Yuan (Google) &middot; Martin Wattenberg (Google) &middot; Fernanda B Viegas (Google) &middot; Andy Coenen (Google) &middot; Adam Pearce (Google) &middot; Been Kim (Google)</i></p>

      <p><b>Self-Critical Reasoning for Robust Visual Question Answering</b><br><i>Jialin Wu (UT Austin) &middot; Raymond Mooney (University of Texas at Austin)</i></p>

      <p><b>Learning to Screen</b><br><i>Alon Cohen (Technion and Google Inc.) &middot; Avinatan Hassidim (Google) &middot; Haim Kaplan (TAU, GOOGLE) &middot; Yishay Mansour (Tel Aviv University / Google) &middot; Shay Moran (IAS, Princeton)</i></p>

      <p><b>A Communication Efficient Stochastic Multi-Block Alternating Direction Method of Multipliers</b><br><i>Hao Yu (Alibaba Group (US) Inc )</i></p>

      <p><b>A Little Is Enough: Circumventing Defenses For Distributed Learning</b><br><i>Gilad Baruch (Bar Ilan University) &middot; Moran Baruch (Bar Ilan University) &middot; Yoav Goldberg (Bar-Ilan University)</i></p>

      <p><b>Error Correcting Output Codes Improve Probability Estimation and Adversarial Robustness of Deep Neural Networks</b><br><i>Gunjan Verma (ARL) &middot; Ananthram Swami (Army Research Laboratory, Adelphi)</i></p>

      <p><b>A Robust Non-Clairvoyant Dynamic Mechanism for Contextual Auctions</b><br><i>Yuan Deng (Duke University) &middot; Sebastien Lahaie (Google Research) &middot; Vahab Mirrokni (Google Research NYC)</i></p>

      <p><b>Finite-Sample Analysis for SARSA with Linear Function Approximation</b><br><i>Shaofeng Zou (University at Buffalo, the State University of New York) &middot; Tengyu Xu (The Ohio State University) &middot; Yingbin Liang (The Ohio State University)</i></p>

      <p><b>Who is Afraid of Big Bad Minima? Analysis of gradient-flow in spiked matrix-tensor models</b><br><i>Stefano Sarao Mannelli (Institut de Physique Théorique) &middot; Giulio Biroli (ENS) &middot; Chiara Cammarota (King's College London) &middot; Florent Krzakala (École Normale Supérieure) &middot; Lenka Zdeborová (CEA Saclay)</i></p>

      <p><b>Graph Structured Prediction Energy Networks</b><br><i>Colin Graber (University of Illinois at Urbana-Champaign) &middot; Alexander Schwing (University of Illinois at Urbana-Champaign)</i></p>

      <p><b>Private Learning Implies Online Learning: An Efficient Reduction</b><br><i>Alon Gonen (Princeton University) &middot; Elad Hazan (Princeton University) &middot; Shay Moran (IAS, Princeton)</i></p>

      <p><b>Graph Agreement Models for Semi-Supervised Learning</b><br><i>Otilia Stretcu (Carnegie Mellon University) &middot; Krishnamurthy Viswanathan (Google Research) &middot; Dana Movshovitz-Attias (Google) &middot; Emmanouil Platanios (Carnegie Mellon University) &middot; Sujith Ravi (Google Research) &middot; Andrew Tomkins (Google)</i></p>

      <p><b>Latent distance estimation for random geometric graphs</b><br><i>Ernesto J Araya Valdivia (Université Paris-Sud) &middot; Yohann De Castro (ENPC)</i></p>

      <p><b>Seeing the Wind: Visual Wind Speed Prediction with a Coupled Convolutional and Recurrent Neural Network</b><br><i>Jennifer Cardona (Stanford University) &middot; Michael Howland (Stanford University) &middot; John Dabiri (Stanford University)</i></p>

      <p><b>The Functional Neural Process</b><br><i>Christos Louizos (University of Amsterdam) &middot; Xiahan Shi (Bosch Center for Artificial Intelligence) &middot; Klamer Schutte (TNO) &middot; Max Welling (University of Amsterdam / Qualcomm AI Research)</i></p>

      <p><b>Recurrent Registration Neural Networks for Deformable Image Registration</b><br><i>Robin Sandkühler (Department of Biomedical Engineering, University of Basel) &middot; Simon Andermatt (Center for medical Image Analysis and Navigation) &middot; Grzegorz Bauman (University of Basel Hospital) &middot; Sylvia  Nyilas (Bern University Hospital) &middot; Christoph Jud (University of Basel) &middot; Philippe C. Cattin (University of Basel)</i></p>

      <p><b>Unsupervised State Representation Learning in Atari</b><br><i>Ankesh Anand (Mila, Université de Montréal) &middot; Evan Racah (Mila, Université de Montréal) &middot; Sherjil Ozair (Université de Montréal) &middot; Yoshua Bengio (Mila) &middot; Marc-Alexandre Côté (Microsoft Research) &middot; R Devon Hjelm (Microsoft Research)</i></p>

      <p><b>Unlocking Fairness: a Trade-off Revisited</b><br><i>Michael Wick (Oracle Labs) &middot; swetasudha panda (Oracle Labs) &middot; Jean-Baptiste Tristan (Oracle Labs)</i></p>

      <p><b>Fisher Efficient Inference of Intractable Models</b><br><i>Song Liu (University of Bristol) &middot; Takafumi Kanamori (Tokyo Institute of Technology/RIKEN) &middot; Wittawat Jitkrittum (Max Planck Institute for Intelligent Systems) &middot; Yu Chen (University of Bristol)</i></p>

      <p><b>Thompson Sampling and Approximate Inference</b><br><i>My Phan (University of Massachusetts Amherst) &middot; Yasin Abbasi (Adobe Research) &middot; Justin Domke (University of Massachusetts, Amherst)</i></p>

      <p><b>PRNet: Self-Supervised Learning for Partial-to-Partial Registration</b><br><i>Yue Wang (MIT) &middot; Justin M Solomon (MIT)</i></p>

      <p><b>Surrogate Objectives for Batch Policy Optimization in One-step Decision Making</b><br><i>Minmin Chen (Google) &middot; Ramki Gummadi (Google) &middot; Chris Harris (Google) &middot; Dale Schuurmans (University of Alberta & Google Brain)</i></p>

      <p><b>Modelling heterogeneous distributions with an Uncountable Mixture of Asymmetric Laplacians</b><br><i>Axel Brando (BBVA Data & Analytics and Universitat de Barcelona) &middot; Jose A Rodriguez (BBVA Data & Analytics) &middot; Jordi Vitria (Universitat de Barcelona) &middot; Alberto Rubio Muñoz (BBVA Data & Analytics)</i></p>

      <p><b>Learning Macroscopic Brain Connectomes via Group-Sparse Factorization</b><br><i>Farzane Aminmansour (University of Alberta) &middot; Andrew Patterson (University of Alberta) &middot; Lei Le (Indiana University Bloomington) &middot; Yisu  Peng (Northeastern University) &middot; Daniel  Mitchell (University of Alberta) &middot; Franco Pestilli (Indiana University) &middot; Cesar Caiafa (CONICET/RIKEN AIP) &middot; Russell Greiner (University of Alberta) &middot; Martha White (University of Alberta)</i></p>

      <p><b>Approximating the Permanent by Sampling from Adaptive Partitions</b><br><i>Jonathan Kuck (Stanford) &middot; Tri Dao (Stanford University) &middot; Hamid Rezatofighi (University of Adelaide) &middot; Ashish Sabharwal (Allen Institute for AI) &middot; Stefano Ermon (Stanford)</i></p>

      <p><b>Retrosynthesis Prediction with Conditional Graph Logic Network</b><br><i>Hanjun Dai (Georgia Tech) &middot; Chengtao Li (MIT) &middot; Connor Coley (MIT) &middot; Bo Dai (Google Brain) &middot; Le Song (Ant Financial & Georgia Institute of Technology)</i></p>

      <p><b>Procrastinating with Confidence: Near-Optimal, Anytime, Adaptive Algorithm Configuration</b><br><i>Robert Kleinberg (Cornell University) &middot; Kevin Leyton-Brown (University of British Columbia) &middot; Brendan Lucier (Microsoft Research) &middot; Devon Graham (University of British Columbia)</i></p>

      <p><b>Online Learning via the Differential Privacy Lens</b><br><i>Jacob Abernethy (Georgia Institute of Technolog) &middot; Young Hun Jung (Universith of Michigan) &middot; Chansoo Lee (University of Michigan) &middot; Audra McMillan (Boston Univ) &middot; Ambuj Tewari (University of Michigan)</i></p>

      <p><b>3D Object Detection from a Single RGB Image via Perspective Points</b><br><i>Siyuan Huang (University of California, Los Angeles) &middot; Yixin Chen (UCLA) &middot; Tao Yuan (UCLA) &middot; Siyuan Qi (UCLA) &middot; Yixin Zhu (University of California, Los Angeles) &middot; Song-Chun Zhu (UCLA)</i></p>

      <p><b>Parameter elimination in particle Gibbs sampling</b><br><i>Anna Wigren (Uppsala University) &middot; Riccardo Sven Risuleo (Uppsala University) &middot; Lawrence Murray (Uber AI Labs) &middot; Fredrik Lindsten (Linköping Universituy)</i></p>

      <p><b>This Looks Like That: Deep Learning for Interpretable Image Recognition</b><br><i>Chaofan Chen (Duke University) &middot; Oscar Li (Duke University) &middot; Chaofan Tao (Duke University) &middot; Alina Barnett (Duke University) &middot; Cynthia Rudin (Duke)</i></p>

      <p><b>Adaptively Aligned Image Captioning via Adaptive Attention Time</b><br><i>Lun Huang (Peking University) &middot; Wenmin Wang (Peking University) &middot; Yaxian Xia (Peking University) &middot; Jie Chen (Peng Cheng Laboratory)</i></p>

      <p><b>Accurate Uncertainty Estimation and Decomposition in Ensemble Learning</b><br><i>Jeremiah Liu (Harvard University) &middot; John Paisley (Columbia University) &middot; Marianthi-Anna Kioumourtzoglou (Columbia University) &middot; Brent Coull (Harvard University)</i></p>

      <p><b>Learning Bayesian Networks with Low Rank Conditional Probability Tables</b><br><i>Adarsh Barik (Purdue University) &middot; Jean Honorio (Purdue University)</i></p>

      <p><b>Equal Opportunity in Online Classification with Partial Feedback</b><br><i>Yahav Bechavod (Hebrew University of Jerusalem) &middot; Katrina Ligett (Hebrew University) &middot; Aaron Roth (University of Pennsylvania) &middot; Bo Waggoner (U. Colorado, Boulder) &middot; Steven Wu (Microsoft Research)</i></p>

      <p><b>Modeling Expectation Violation in Intuitive Physics with Coarse Probabilistic Object Representations</b><br><i>Kevin Smith (MIT) &middot; Lingjie Mei (MIT) &middot; Shunyu Yao (Princeton University) &middot; Jiajun Wu (MIT) &middot; Elizabeth Spelke (Harvard University) &middot; Josh Tenenbaum (MIT) &middot; Tomer Ullman (MIT)</i></p>

      <p><b>Neural Multisensory Scene Inference</b><br><i>Jae Hyun Lim (MILA, University of Montreal) &middot; Pedro O. Pinheiro (Element AI) &middot; Negar Rostamzadeh (Elemenet AI) &middot; Chris Pal (MILA, Polytechnique Montréal, Element AI) &middot; Sungjin Ahn (Rutgers University)</i></p>

      <p><b>Regret Bounds for Thompson Sampling in Restless Bandit Problems</b><br><i>Young Hun Jung (Universith of Michigan) &middot; Ambuj Tewari (University of Michigan)</i></p>

      <p><b>What Can ResNet Learn Efficiently, Going Beyond Kernels?</b><br><i>Zeyuan Allen-Zhu (Microsoft Research) &middot; Yuanzhi Li (Princeton)</i></p>

      <p><b>Better Transfer Learning Through Inferred Successor Maps</b><br><i>Tamas Madarasz (University of Oxford) &middot; Tim Behrens (University of Oxford)</i></p>

      <p><b>Unsupervised Co-Learning on $G$-Manifolds Across Irreducible Representations</b><br><i>Yifeng Fan (University of Illinois at Urbana-Champaign) &middot; Tingran Gao (University of Chicago) &middot; Jane Zhao (University of Illinois at Urbana Champaign)</i></p>

      <p><b>Defending Against Neural Fake News</b><br><i>Rowan Zellers (University of Washington) &middot; Ari Holtzman (University of Washington) &middot; Hannah Rashkin (University of Washington) &middot; Yonatan Bisk (University of Washington) &middot; Ali Farhadi (University of Washington, Allen Institute for Artificial Intelligence) &middot; Franziska Roesner (University of Washington) &middot; Yejin Choi (University of Washington)</i></p>

      <p><b>Sample Adaptive MCMC</b><br><i>Michael Zhu (Stanford University)</i></p>

      <p><b>A Stochastic Composite Gradient Method with Incremental Variance Reduction</b><br><i>Junyu Zhang (University of Minnesota) &middot; Lin Xiao (Microsoft Research)</i></p>

      <p><b>Nonparametric Density Estimation &amp; Convergence Rates for GANs under Besov IPM Losses</b><br><i>Ananya Uppal (Carnegie Mellon University) &middot; Shashank Singh (Carnegie Mellon University) &middot; Barnabas Poczos (Carnegie Mellon University)</i></p>

      <p><b>STAR-Caps: Capsule Networks with Straight-Through Attentive Routing</b><br><i>Karim Ahmed (Dartmouth) &middot; Lorenzo Torresani (Facebook)</i></p>

      <p><b>Limitations of Lazy Training of Two-layers Neural Network</b><br><i>Song Mei (Stanford University) &middot; Theodor Misiakiewicz (Stanford University) &middot; Behrooz Ghorbani (Stanford University) &middot; Andrea Montanari (Stanford)</i></p>

      <p><b>Reconciling meta-learning and continual learning with online mixtures of tasks</b><br><i>Ghassen Jerfel (Duke University) &middot; Erin Grant (UC Berkeley) &middot; Thomas Griffiths (Princeton University) &middot; Katherine Heller (Google)</i></p>

      <p><b>Distributionally Robust Optimization and Generalization in Kernel Methods</b><br><i>Matthew Staib (MIT) &middot; Stefanie Jegelka (MIT)</i></p>

      <p><b>A General Theory of Equivariant CNNs on Homogeneous Spaces</b><br><i>Taco S Cohen (Qualcomm AI Research) &middot; Mario Geiger (EPFL) &middot; Maurice Weiler (University of Amsterdam)</i></p>

      <p><b>Trivializations for Gradient-Based Optimization on Manifolds</b><br><i>Mario Lezcano Casado (Univeristy of Oxford)</i></p>

      <p><b>Write, Execute, Assess: Program Synthesis with a REPL</b><br><i>Kevin Ellis (MIT) &middot; Maxwell Nye (MIT) &middot; Yewen Pu (MIT) &middot; Felix Sosa (Harvard) &middot; Josh Tenenbaum (MIT) &middot; Armando Solar-Lezama (MIT)</i></p>

      <p><b>(Nearly) Efficient Algorithms for the Graph Matching Problem on Correlated Random Graphs</b><br><i>Boaz Barak (Harvard University) &middot; Chi-Ning Chou (Harvard University) &middot; Zhixian Lei (Harvard University) &middot; Tselil Schramm (Harvard University) &middot; Yueqi Sheng (Harvard University )</i></p>

      <p><b>Preference-Based Batch and Sequential Teaching: Towards a Unified View of Models</b><br><i>Farnam Mansouri (Max Planck Institute for Software Systems) &middot; Yuxin Chen (Caltech) &middot; Ara Vartanian (University of Wisconsin -- Madison) &middot; Jerry Zhu (University of Wisconsin-Madison) &middot; Adish Singla (MPI-SWS)</i></p>

      <p><b>Online Continuous Submodular Maximization: From Full-Information to Bandit Feedback</b><br><i>Mingrui Zhang (Yale University) &middot; Lin Chen (Yale University) &middot; Hamed Hassani (UPenn) &middot; Amin Karbasi (Yale)</i></p>

      <p><b>Sampling Networks and Aggregate Simulation for Online POMDP Planning</b><br><i>Hao Cui (Tufts University) &middot; Roni Khardon (Indiana University, Bloomington)</i></p>

      <p><b>Correlation in Extensive-Form Games: Saddle-Point Formulation and Benchmarks</b><br><i>Gabriele Farina (Carnegie Mellon University) &middot; Chun Kai Ling (Carnegie Mellon University) &middot; Fei Fang (Carnegie Mellon University) &middot; Tuomas Sandholm (Carnegie Mellon University)</i></p>

      <p><b>GNNExplainer: Generating Explanations for Graph Neural Networks</b><br><i>Zhitao Ying (Stanford University) &middot; Dylan Bourgeois (EPFL) &middot; Jiaxuan You (Stanford University) &middot; Marinka Zitnik (Stanford University) &middot; Jure Leskovec (Stanford University and Pinterest)</i></p>

      <p><b>Linear Stochastic Bandits Under Safety Constraints</b><br><i>Sanae Amani (University of California Santa Barbara) &middot; Mahnoosh Alizadeh (University of California Santa Barbara) &middot; Christos Thrampoulidis (UCSB)</i></p>

      <p><b>A coupled autoencoder approach for multi-modal analysis of cell types</b><br><i>Rohan Gala (Allen Institute) &middot; Nathan Gouwens (Allen Institute) &middot; Zizhen Yao (Allen Institute) &middot; Agata Budzillo (Allen Institute) &middot; Osnat Penn (Allen Institute) &middot; Bosiljka Tasic (Allen Institute) &middot; Gabe Murphy (Allen Institute) &middot; Hongkui Zeng (Allen Institute) &middot; Uygar Sumbul (Allen Institute)</i></p>

      <p><b>Towards Automatic Concept-based Explanations</b><br><i>Amirata Ghorbani (Stanford University) &middot; James Wexler () &middot; James Zou (Stanford University) &middot; Been Kim (Google)</i></p>

      <p><b>A Deep Probabilistic Model for Compressing Low Resolution Videos</b><br><i>Salvator Lombardo (Disney Research) &middot; JUN HAN (Dartmouth College) &middot; Christopher Schroers (Disney Research) &middot; Stephan Mandt (Disney Research)</i></p>

      <p><b>Budgeted Reinforcement Learning in Continuous State Space</b><br><i>Nicolas Carrara (inria) &middot; Edouard Leurent (INRIA) &middot; Romain Laroche (Microsoft Research) &middot; Tanguy Urvoy (Orange-Labs) &middot; Odalric-Ambrym Maillard (INRIA) &middot; Olivier Pietquin (Google Research    Brain Team)</i></p>

      <p><b>The Discovery of Useful Questions as Auxiliary Tasks</b><br><i>Vivek Veeriah (University of Michigan) &middot; Richard L Lewis (University of Michigan) &middot; Janarthanan Rajendran (University of Michigan) &middot; David Silver (DeepMind) &middot; Satinder Singh (University of Michigan)</i></p>

      <p><b>Sinkhorn Barycenters with Free Support via Frank-Wolfe Algorithm</b><br><i>Giulia Luise (University College London) &middot; Saverio Salzo (Istituto Italiano di Tecnologia) &middot; Massimiliano Pontil (IIT & UCL) &middot; Carlo Ciliberto (Imperial College London)</i></p>

      <p><b>Finding the Needle in the Haystack with Convolutions: on the benefits of architectural bias</b><br><i>Stéphane d'Ascoli (ENS) &middot; Levent Sagun (EPFL) &middot; Giulio Biroli (ENS) &middot; Joan Bruna (NYU)</i></p>

      <p><b>Correlation clustering with local objectives</b><br><i>Sanchit Kalhan (Northwestern University) &middot; Konstantin Makarychev (Northwestern University) &middot;  Timothy Zhou (Northwestern University)</i></p>

      <p><b>Multiclass Performance Metric Elicitation</b><br><i>Gaurush Hiranandani (UNIVERSITY OF ILLINOIS, URBANA-CH) &middot; Shant Boodaghians (UIUC) &middot; Ruta Mehta (UIUC) &middot; Oluwasanmi Koyejo (UIUC)</i></p>

      <p><b>Algorithmic Analysis and Statistical Estimation of SLOPE via Approximate Message Passing</b><br><i>Zhiqi Bu (University of Pennsylvania) &middot; Jason Klusowski (Rutgers University) &middot; Cynthia Rush (Columbia University) &middot; Weijie Su (University of Pennsylvania)</i></p>

      <p><b>Explicit Explore-Exploit Algorithms in Continuous State Spaces</b><br><i>Mikael Henaff (NYU)</i></p>

      <p><b>ADDIS: an adaptive discarding algorithm for online FDR control with conservative nulls</b><br><i>Jinjin Tian (Carnegie Mellon University) &middot; Aaditya Ramdas (Carnegie Mellon University)</i></p>

      <p><b>Slice-based Learning: A Programming Model for Residual Learning in Critical Data Slices</b><br><i>Vincent Chen (Stanford University) &middot; Sen Wu (Stanford University) &middot; Alexander Ratner (Stanford) &middot; Jen Weng (Stanford University) &middot; Christopher Ré (Stanford)</i></p>

      <p><b>Understanding Posterior Collapse in Variational Autoencoders</b><br><i>James Lucas (University of Toronto) &middot; George Tucker (Google Brain) &middot; Roger Grosse (University of Toronto) &middot; Mohammad Norouzi (Google Brain)</i></p>

      <p><b>Language as an Abstraction for Hierarchical Deep Reinforcement Learning</b><br><i>YiDing Jiang (Google) &middot; Shixiang (Shane) Gu (Google Brain) &middot; Kevin P Murphy (Google) &middot; Chelsea Finn (Google Brain)</i></p>

      <p><b>Efficient online learning with kernels for adversarial large scale problems</b><br><i>Rémi Jézéquel (INRIA - Paris) &middot; Pierre Gaillard () &middot; Alessandro Rudi (INRIA, Ecole Normale Superieure)</i></p>

      <p><b>A Linearly Convergent Method for Non-Smooth Non-Convex Optimization on the Grassmannian with Applications to Robust Subspace and Dictionary Learning</b><br><i>Zhihui Zhu (Johns Hopkins University) &middot; Tianyu Ding (Johns Hopkins University) &middot; Daniel Robinson (Johns Hopkins University) &middot; Manolis Tsakiris (ShanghaiTech University) &middot; Rene Vidal (Johns Hopkins University)</i></p>

      <p><b>ObjectNet: A large-scale bias-controlled dataset for pushing the limits of object recognition models</b><br><i>Andrei Barbu (MIT) &middot; David Mayo (MIT) &middot; Julian Alverio (MIT) &middot; William Luo (MIT) &middot; Christopher Wang (Massachusetts Institute of Technology) &middot; Dan Gutfreund (IBM Research) &middot; Josh Tenenbaum (MIT) &middot; Boris Katz (MIT)</i></p>

      <p><b>Certified Adversarial Robustness with Addition Gaussian Noise</b><br><i>Bai Li (Duke University) &middot; Changyou Chen (University at Buffalo) &middot; Wenlin Wang (Duke Univeristy) &middot; Lawrence Carin (Duke University)</i></p>

      <p><b>Tight Dimensionality Reduction for Sketching Low Degree Polynomial Kernels</b><br><i>Michela Meister (Google) &middot; Tamas Sarlos (Google Research) &middot; David Woodruff (Carnegie Mellon University)</i></p>

      <p><b>Non-Cooperative Inverse Reinforcement Learning</b><br><i>Xiangyuan Zhang (University of Illinois at Urbana-Champaign) &middot; Kaiqing Zhang (University of Illinois at Urbana-Champaign (UIUC)) &middot; Erik Miehling (University of Illinois at Urbana-Champaign) &middot; Tamer Basar ()</i></p>

      <p><b>DINGO: Distributed Newton-Type Method for Gradient-Norm Optimization</b><br><i>Rixon Crane (The University of Queensland) &middot; Farbod Roosta-Khorasani (University of Queensland)</i></p>

      <p><b>Sobolev Independence Criterion </b><br><i>Youssef Mroueh (IBM T.J Watson Research Center) &middot; Tom Sercu (IBM Research AI) &middot; Mattia Rigotti (IBM Research AI) &middot; Inkit Padhi (IBM Research) &middot; Cicero Nogueira dos Santos (IBM Research)</i></p>

      <p><b>Maximum Entropy Monte-Carlo Planning</b><br><i>Chenjun Xiao (University of Alberta) &middot; Ruitong Huang (Borealis AI) &middot; Jincheng Mei (University of Alberta) &middot; Dale Schuurmans (Google) &middot; Martin Müller (University of Alberta)</i></p>

      <p><b>Learning from brains how to regularize machines</b><br><i>Zhe Li (Baylor College of Medicine) &middot; Wieland Brendel (AG Bethge, University of Tübingen) &middot; Edgar Walker (Baylor College of Medicine) &middot; Erick Cobos (Baylor College of Medicine) &middot; Taliah Muhammad (Baylor College of Medicine) &middot; Jacob Reimer (Baylor College of Medicine) &middot; Matthias Bethge (University of Tübingen) &middot; Fabian Sinz (University Tübingen) &middot; Zachary Pitkow (BCM/Rice) &middot; Andreas Tolias (Baylor College of Medicine)</i></p>

      <p><b>Using Statistics to Automate Stochastic Optimization</b><br><i>Hunter Lang (Microsoft Research) &middot; Lin Xiao (Microsoft Research) &middot; Pengchuan Zhang (Microsoft Research)</i></p>

      <p><b>Zero-shot Knowledge Transfer via Adversarial Belief Matching</b><br><i>Paul Micaelli (The University of Edinburgh) &middot; Amos Storkey (University of Edinburgh)</i></p>

      <p><b>Differentiable Convex Optimization Layers</b><br><i>Akshay Agrawal (Stanford University) &middot; Brandon Amos (Facebook) &middot; Shane Barratt (Stanford University) &middot; Stephen Boyd (Stanford University) &middot; Steven Diamond (Stanford University) &middot; J. Zico Kolter (Carnegie Mellon University / Bosch Center for AI)</i></p>

      <p><b>Random Tessellation Forests</b><br><i>Shufei Ge (Simon Fraser University) &middot; Shijia Wang (Simon Fraser University) &middot; Yee Whye Teh (University of Oxford, DeepMind) &middot; Liangliang Wang (Simon Fraser University) &middot; Lloyd T Elliott (Simon Fraser University)</i></p>

      <p><b>Learning Nearest Neighbor Graphs from Noisy Distance Samples</b><br><i>Blake Mason (University of Wisconsin - Madison) &middot; Ardhendu Tripathy (University of Wisconsin - Madison) &middot; Robert Nowak (University of Wisconsion-Madison)</i></p>

      <p><b>Lookahead Optimizer: k steps forward, 1 step back</b><br><i>Michael Zhang (University of Toronto) &middot; James Lucas (University of Toronto) &middot; Jimmy Ba (University of Toronto / Vector Institute) &middot; Geoffrey Hinton (Google)</i></p>

      <p><b>Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer</b><br><i>Wenzheng Chen (University of Toronto) &middot; Huan Ling (University of Toronto, NVIDIA) &middot; Jun Gao (University of Toronto) &middot; Edward Smith (McGill University) &middot; Jaakko Lehtinen (NVIDIA Research; Aalto University) &middot; Alec Jacobson (University of Toronto) &middot; Sanja Fidler (University of Toronto)</i></p>

      <p><b>Covariate-Powered Empirical Bayes Estimation</b><br><i>Nikolaos Ignatiadis (Stanford University) &middot; Stefan Wager (Stanford University)</i></p>

      <p><b>Understanding the Role of Momentum in Stochastic Gradient Methods</b><br><i>Igor Gitman (Microsoft Research AI) &middot; Hunter Lang (Microsoft Research) &middot; Pengchuan Zhang (Microsoft Research) &middot; Lin Xiao (Microsoft Research)</i></p>

      <p><b>A neurally plausible model for online recognition andpostdiction in a dynamical environment</b><br><i>Li Wenliang (Gatsby Unit, UCL) &middot; Maneesh Sahani (Gatsby Unit, UCL)</i></p>

      <p><b>Guided Meta-Policy Search</b><br><i>Russell Mendonca (UC Berkeley) &middot; Abhishek Gupta (University of California, Berkeley) &middot; Rosen Kralev (UC Berkeley) &middot; Pieter Abbeel (UC Berkeley  Covariant) &middot; Sergey Levine (UC Berkeley) &middot; Chelsea Finn (Stanford University)</i></p>

      <p><b>Marginalized Off-Policy Evaluation for Reinforcement Learning</b><br><i>Tengyang Xie (University of Illinois at Urbana-Champaign) &middot; Yifei Ma (Amazon) &middot; Yu-Xiang Wang (UC Santa Barbara)</i></p>

      <p><b>Contextual Bandits with Cross-Learning</b><br><i>Santiago Balseiro (Columbia University) &middot; Negin Golrezaei (University of Southern California) &middot; Mohammad Mahdian (Google Research) &middot; Vahab Mirrokni (Google Research NYC) &middot; Jon Schneider (Google Research)</i></p>

      <p><b>Evaluating Protein Transfer Learning with TAPE</b><br><i>Roshan Rao (UC Berkeley) &middot; Nicholas Bhattacharya (UC Berkeley) &middot; Neil Thomas (UC Berkeley) &middot; Yan Duan (COVARIANT.AI) &middot; Peter Chen (COVARIANT.AI) &middot; John Canny (UC Berkeley) &middot; Pieter Abbeel (UC Berkeley  Covariant) &middot; Yun Song (UC Berkeley)</i></p>

      <p><b>A Bayesian Theory of Conformity in Collective Decision Making</b><br><i>Koosha Khalvati (University of Washington) &middot; Saghar Mirbagheri (New York University) &middot; Seongmin A. Park (Cognitive Neuroscience Center, CNRS) &middot; Jean-Claude Dreher (cnrs) &middot; Rajesh PN Rao (University of Washington)</i></p>

      <p><b>Regularization Matters: Generalization and Optimization of Neural Nets v.s. their Induced Kernel</b><br><i>Colin Wei (Stanford University) &middot; Jason Lee (USC) &middot; Qiang Liu (UT Austin) &middot; Tengyu Ma (Stanford)</i></p>

      <p><b>Data-dependent Sample Complexity of Deep Neural Networks via Lipschitz Augmentation</b><br><i>Colin Wei (Stanford University) &middot; Tengyu Ma (Stanford)</i></p>

      <p><b>A Benchmark for Interpretability Methods in Deep Neural Networks</b><br><i>Sara Hooker (Google AI Resident) &middot; Dumitru Erhan (Google Brain) &middot; Pieter-Jan Kindermans (Google Brain) &middot; Been Kim (Google)</i></p>

      <p><b>Memory Efficient Adaptive Optimization</b><br><i>Rohan Anil (Google) &middot; Vineet Gupta (Google) &middot; Tomer Koren (Google) &middot; Yoram Singer (Google)</i></p>

      <p><b>Dynamic Incentive-Aware Learning: Robust Pricing in Contextual Auctions</b><br><i>Negin Golrezaei (MIT) &middot; Adel Javanmard (USC) &middot; Vahab Mirrokni (Google Research NYC)</i></p>

      <p><b>Convergence-Rate-Matching Discretization of Accelerated Optimization Flows Through Opportunistic State-Triggered Control</b><br><i>Miguel Vaquero (UCSD) &middot; Jorge Cortes (UCSD)</i></p>

      <p><b>A Unified Framework for Data Poisoning Attack to Graph-based Semi-supervised Learning</b><br><i>Xuanqing Liu (University of California, Los Angeles) &middot; Si Si (Google Research) &middot; Jerry Zhu (University of Wisconsin-Madison) &middot; Yang Li (Google) &middot; Cho-Jui Hsieh (UCLA)</i></p>

      <p><b>Systematic generalization through meta sequence-to-sequence learning</b><br><i>Brenden Lake (New York University)</i></p>

      <p><b>Bayesian Joint Estimation of Multiple Graphical Models</b><br><i>Lingrui Gan (University of Illinois at Urbana and Champaign) &middot; Xinming Yang (University of Illinois at Urbana-Champaign) &middot; Naveen Narisetty (University of Illinois at Urbana-Champaign) &middot; Feng Liang (Univ. of Illinois Urbana-Champaign Statistics)</i></p>

      <p><b>Practical Two-Step Lookahead Bayesian Optimization</b><br><i>Jian Wu (Cornell University) &middot; Peter Frazier (Cornell / Uber)</i></p>

      <p><b>Leader Stochastic Gradient Descent for Distributed Training of Deep Learning Models</b><br><i>Yunfei Teng (New York University) &middot; Wenbo Gao (Columbia University) &middot; François Chalus (Credit Suisse) &middot; Anna Choromanska (NYU) &middot; Donald Goldfarb (Columbia University) &middot; Adrian Weller (Cambridge, Alan Turing Institute)</i></p>

      <p><b>A Convex Relaxation Barrier to Tight Robustness Verification of Neural Networks</b><br><i>Hadi Salman (Microsoft Research AI) &middot; Greg Yang (Microsoft Research) &middot; Huan Zhang (UCLA) &middot; Cho-Jui Hsieh (UCLA) &middot; Pengchuan Zhang (Microsoft Research)</i></p>

      <p><b>Neural Jump Stochastic Differential Equations</b><br><i>Junteng Jia (Cornell) &middot; Austin Benson (Cornell University)</i></p>

      <p><b>Learning metrics for persistence-based summaries and applications for graph classification</b><br><i>Qi Zhao (The Ohio State University) &middot; Yusu Wang (Ohio State University)</i></p>

      <p><b>ON THE VALUE OF TARGET SAMPLING IN COVARIATE-SHIFT</b><br><i>Steve Hanneke (Toyota Technological Institute at Chicago) &middot; Samory Kpotufe (Columbia University)</i></p>

      <p><b>Stochastic Variance Reduced Primal Dual Algorithms for Empirical Composition Optimization</b><br><i>Adithya M Devraj (University of Florida ) &middot; Jianshu Chen (Tencent AI Lab)</i></p>

      <p><b>On Robustness of Principal Component Regression</b><br><i>Anish Agarwal (MIT) &middot; Devavrat Shah (Massachusetts Institute of Technology) &middot; Dennis Shen (Massachusetts Institute of Technology) &middot; Dogyoon Song (Massachusetts Institute of Technology)</i></p>

      <p><b>Meta Learning with Relational Information for Short Sequences</b><br><i>Yujia Xie (Georgia Institute of Technology) &middot; Haoming Jiang (Georgia Institute of Technology) &middot; Feng Liu (Florida Atlantic University) &middot; Tuo Zhao (Georgia Tech) &middot; Hongyuan Zha (Georgia Tech)</i></p>

      <p><b>Residual Flows for Invertible Generative Modeling</b><br><i>Tian Qi Chen (U of Toronto) &middot; Jens Behrmann (University of Bremen) &middot; David Duvenaud (University of Toronto) &middot; Joern-Henrik Jacobsen (Vector Institute)</i></p>

      <p><b>Multi-Agent Common Knowledge Reinforcement Learning</b><br><i>Christian Schroeder (University of Oxford) &middot; Jakob Foerster (University of Oxford) &middot; Gregory Farquhar (University of Oxford) &middot; Philip Torr (University of Oxford) &middot; Wendelin Boehmer (University of Oxford) &middot; Shimon Whiteson (University of Oxford)</i></p>

      <p><b>Learning to Learn By Self-Critique</b><br><i>Antreas Antoniou (University of Edinburgh) &middot; Amos Storkey (University of Edinburgh)</i></p>

      <p><b>Wide Feedforward or Recurrent Neural Networks of Any Architecture are Gaussian Processes</b><br><i>Greg Yang (Microsoft Research)</i></p>

      <p><b>Neural Networks with Cheap Differential Operators</b><br><i>Tian Qi Chen (U of Toronto) &middot; David Duvenaud (University of Toronto)</i></p>

      <p><b>Transductive Zero-Shot Learning with Visual Structure Constraint</b><br><i>Ziyu Wan (City University of Hong Kong) &middot; Dongdong Chen (university of science and technology of china) &middot; Yan Li (Institute of Automation, Chinese Academy of Sciences) &middot; Xingguang Yan (Shenzhen University) &middot; Junge Zhang (CASIA) &middot; Yizhou Yu (Deepwise AI Lab) &middot; Jing Liao (City University of Hong Kong)</i></p>

      <p><b>Dying Experts: Efficient Algorithms with Optimal Regret Bounds</b><br><i>Hamid Shayestehmanesh (University of Victoria) &middot; Sajjad Azami (University of Victoria) &middot; Nishant Mehta (University of Victoria)</i></p>

      <p><b>Model similarity mitigates test set overuse</b><br><i>Horia Mania (UC Berkeley) &middot; John Miller (University of California, Berkeley) &middot; Ludwig Schmidt (UC Berkeley) &middot; Moritz Hardt (University of California, Berkeley) &middot; Benjamin Recht (UC Berkeley)</i></p>

      <p><b>A unified theory for the origin of grid cells through the lens of pattern formation</b><br><i>Ben Sorscher (Stanford University) &middot; Gabriel Mel (Stanford University) &middot; Surya Ganguli (Stanford) &middot; Samuel Ocko (Stanford)</i></p>

      <p><b>On Sample Complexity Upper and Lower Bounds for Exact Ranking from Noisy Comparisons</b><br><i>Wenbo Ren (The Ohio State University) &middot; Jia Liu (Iowa State University) &middot; Ness Shroff (The Ohio State University)</i></p>

      <p><b>Hierarchical Decision Making by Generating and Following Natural Language Instructions</b><br><i>Hengyuan Hu (Facebook) &middot; Denis Yarats (New York University) &middot; Qucheng Gong (Facebook AI Research) &middot; Yuandong Tian (Facebook AI Research) &middot; Mike Lewis (Facebook)</i></p>

      <p><b>SHE: A Fast and Accurate Deep Neural Network for Encrypted Data</b><br><i>Qian Lou (Indiana University Bloomington) &middot; Lei Jiang (Indiana University Bloomington)</i></p>

      <p><b>Locality-Sensitive Hashing for f-Divergences: Mutual Information Loss and Beyond</b><br><i>Lin Chen (Yale University) &middot; Hossein Esfandiari (Google Research) &middot; Gang Fu (Google Inc) &middot; Vahab Mirrokni (Google Research NYC)</i></p>

      <p><b>A Game Theoretic Approach to Class-wise Selective Rationalization</b><br><i>Shiyu Chang (IBM T.J. Watson Research Center) &middot; Yang Zhang (IBM T. J. Watson Research) &middot; Mo Yu (IBM Research) &middot; Tommi Jaakkola (MIT)</i></p>

      <p><b>Efficiently avoiding saddle points with zero order methods: No gradients required </b><br><i>Emmanouil Vlatakis-Gkaragkounis (Columbia University) &middot; Lampros Flokas (Columbia University) &middot; Georgios Piliouras (Singapore University of Technology and Design)</i></p>

      <p><b>Metamers of neural networks reveal divergence from human perceptual systems</b><br><i>Jenelle Feather (MIT) &middot; Alex Durango (MIT) &middot; Ray Gonzalez (MIT) &middot; Josh McDermott (Massachusetts Institute of Technology)</i></p>

      <p><b>Spatial-Aware Feature Aggregation for Image based Cross-View Geo-Localization</b><br><i>Yujiao Shi (ANU) &middot; Liu Liu (ANU) &middot; Xin Yu (Australian National University) &middot; Hongdong Li (Australian National University)</i></p>

      <p><b>Decentralized sketching of low rank matrices</b><br><i>Rakshith Sharma (Georgia Tech) &middot; Kiryung Lee (Ohio state university) &middot; Marius  Junge (University of Illinois) &middot; Justin Romberg (Georgia Institute of Technology)</i></p>

      <p><b>Average Case Column Subset Selection for Entrywise $\ell_1$-Norm Loss</b><br><i>Zhao Song (University of Washington) &middot; David Woodruff (Carnegie Mellon University) &middot; Peilin Zhong (Columbia University)</i></p>

      <p><b>Efficient Forward Architecture Search</b><br><i>Hanzhang Hu (Carnegie Mellon University) &middot; John Langford (Microsoft Research New York) &middot; Rich Caruana (Microsoft) &middot; Saurajit Mukherjee (microsoft) &middot; Eric J Horvitz (Microsoft Research) &middot; Debadeepta Dey (Microsoft Research AI)</i></p>

      <p><b>Unsupervised Meta Learning for Few-Show Image Classification</b><br><i>Siavash Khodadadeh (University of Central Florida) &middot; Ladislau Boloni (University of Central Florida) &middot; Mubarak Shah (University of Central Florida)</i></p>

      <p><b>Learning Mixtures of Plackett-Luce Models from Structured Partial Orders</b><br><i>Zhibing Zhao (RPI) &middot; Lirong Xia (RPI)</i></p>

      <p><b>Certainty Equivalence is Efficient for Linear Quadratic Control</b><br><i>Horia Mania (UC Berkeley) &middot; Stephen Tu (UC Berkeley) &middot; Benjamin Recht (UC Berkeley)</i></p>

      <p><b>Scalable Bayesian inference of dendritic voltage via spatiotemporal recurrent state space models</b><br><i>Ruoxi Sun (Columbia University) &middot;  Ian  Kinsella (Columbia University) &middot; Scott Linderman (Columbia University) &middot; Liam Paninski (Columbia University)</i></p>

      <p><b>Logarithmic Regret for Online Control</b><br><i>Naman Agarwal (Google) &middot; Elad Hazan (Princeton University) &middot; Karan Singh (Princeton University)</i></p>

      <p><b>Elliptical Perturbations for Differential Privacy</b><br><i>Matthew Reimherr (Penn State University) &middot; Jordan Awan (Penn State University)</i></p>

      <p><b>Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks</b><br><i>Yaqin Zhou (Nanyang Technological University) &middot; Shangqing Liu (Nanyang Technological University) &middot; Jingkai Siow (Nanyang Technological University) &middot; Xiaoning Du (Nanyang Technological University) &middot; Yang Liu (Nanyang Technology University, Singapore)</i></p>

      <p><b>KNG: The K-Norm Gradient Mechanism</b><br><i>Matthew Reimherr (Penn State University) &middot; Jordan Awan (Penn State University)</i></p>

      <p><b>CXPlain: Causal Explanations for Model Interpretation under Uncertainty</b><br><i>Patrick Schwab (ETH Zurich) &middot; Walter Karlen (ETH Zurich)</i></p>

      <p><b>Regularized Anderson Acceleration for Off-Policy Deep Reinforcement Learning</b><br><i>Wenjie Shi (Tsinghua University) &middot; Shiji Song (Department of Automation, Tsinghua University) &middot; Hui Wu (Tsinghua University) &middot; Ya-Chu Hsu (Tsinghua University) &middot; Cheng Wu (Tsinghua) &middot; Gao Huang (Tsinghua)</i></p>

      <p><b>STREETS: A Novel Camera Network Dataset for Traffic Flow</b><br><i>Corey Snyder (University of Illinois at Urbana-Champaign) &middot; Minh Do (University of Illinois)</i></p>

      <p><b>Sequential Neural Processes</b><br><i>Gautam Singh (Rutgers Univerity) &middot; Jaesik Yoon (SAP) &middot; Youngsung Son (Electronics and Telecommunications Research Institute) &middot; Sungjin Ahn (Rutgers University)</i></p>

      <p><b>Policy Continuation with Hindsight Inverse Dynamics</b><br><i>Hao Sun (CUHK) &middot; Zhizhong Li (The Chinese University of Hong Kong) &middot; Xiaotong Liu (Peking Uinversity) &middot; Bolei Zhou (CUHK) &middot; Dahua Lin (The Chinese University of Hong Kong)</i></p>

      <p><b>Learning to Self-Train for Semi-Supervised Few-Shot Classification</b><br><i>Xinzhe Li (SJTU) &middot; Qianru Sun (National University of Singapore) &middot; Yaoyao Liu (Tianjin University) &middot; Qin Zhou (Alibaba Group) &middot; Shibao Zheng (SJTU) &middot; Tat-Seng Chua (National Univ. of Singapore) &middot; Bernt Schiele (Max Planck Institute for Informatics)</i></p>

      <p><b>Temporal FiLM: Capturing Long-Range Sequence Dependencies with Feature-Wise Modulations.</b><br><i>Sawyer Birnbaum (Stanford University) &middot; Volodymyr Kuleshov (Stanford University / Afresh) &middot; Zayd Enam (Stanford) &middot; Pang Wei W Koh (Stanford University) &middot; Stefano Ermon (Stanford)</i></p>

      <p><b>From Complexity to Simplicity: Adaptive ES-Active Subspaces for Blackbox Optimization</b><br><i>Krzysztof M Choromanski (Google Brain Robotics) &middot; Aldo Pacchiano (UC Berkeley) &middot; Jack Parker-Holder (Columbia University) &middot; Yunhao Tang (Columbia University) &middot; Vikas Sindhwani (Google)</i></p>

      <p><b>On the Expressive Power of Deep Polynomial Neural Networks</b><br><i>Joe Kileel (Princeton University) &middot; Matthew Trager (NYU) &middot; Joan Bruna (NYU)</i></p>

      <p><b>DETOX: A Redundancy-based Framework for Faster and More Robust Gradient Aggregation</b><br><i>Shashank Rajput (University of Wisconsin - Madison) &middot; Hongyi Wang (University of Wisconsin-Madison) &middot; Zachary Charles (University of Wisconsin - Madison) &middot; Dimitris Papailiopoulos (University of Wisconsin-Madison)</i></p>

      <p><b>Can SGD Learn Recurrent Neural Networks with Provable Generalization?</b><br><i>Zeyuan Allen-Zhu (Microsoft Research) &middot; Yuanzhi Li (Princeton)</i></p>

      <p><b>Limits of Private Learning with Access to Public Data</b><br><i>Raef Bassily (The Ohio State University) &middot; Shay Moran (IAS, Princeton) &middot; Noga  Alon (Princeton)</i></p>

      <p><b>Discrete Object Generation with Reversible Inductive Construction</b><br><i>Ari Seff (Princeton University) &middot; Wenda Zhou (Columbia University) &middot; Farhan Damani (Princeton University) &middot; Abigail Doyle (Princeton University) &middot; Ryan Adams (Princeton University)</i></p>

      <p><b>Efficient Near-Optimal Testing of Community Changes in Balanced Stochastic Block Models</b><br><i>Aditya Gangrade (Boston University) &middot; Praveen Venkatesh (Carnegie Mellon University) &middot; Bobak Nazer (Boston University) &middot; Venkatesh Saligrama (Boston University)</i></p>

      <p><b>Keeping Your Distance: Solving Sparse Reward Tasks Using Self-Balancing Shaped Rewards</b><br><i>Alexander Trott (Salesforce Research) &middot; Stephan Zheng (Salesforce) &middot; Caiming Xiong (Salesforce) &middot; Richard Socher (Salesforce)</i></p>

      <p><b>Superset Technique for Approximate Recovery in One-Bit Compressed Sensing</b><br><i>Larkin H Flodin (University of Massachusetts Amherst) &middot; Venkata Gandikota (University of Massachusetts, Amherst) &middot; Arya Mazumdar (University of Massachusetts Amherst)</i></p>

      <p><b>Bandits with Feedback Graphs and Switching Costs</b><br><i>Raman Arora (Johns Hopkins University) &middot; Teodor Vanislavov Marinov (Johns Hopkins University) &middot; Mehryar Mohri (Courant Inst. of Math. Sciences & Google Research)</i></p>

      <p><b>Functional Adversarial Attacks</b><br><i>Cassidy Laidlaw (University of Maryland) &middot; Soheil Feizi (University of Maryland, College Park)</i></p>

      <p><b>Statistical-Computational Tradeoff in Single Index Models</b><br><i>Lingxiao Wang (Northwestern University) &middot; Zhuoran Yang (Princeton University) &middot; Zhaoran Wang (Northwestern University)</i></p>

      <p><b>On Fenchel Mini-Max Learning</b><br><i>Chenyang Tao (Duke University) &middot; Liqun Chen (Duke University) &middot; Shuyang Dai (Duke University) &middot; Junya Chen (Duke U) &middot; Ke Bai (Duke University) &middot; Dong Wang (Duke University) &middot; Jianfeng Feng (Fudan University) &middot; Wenlian Lu (Fudan University) &middot; Georgiy Bobashev (RTI International) &middot; Lawrence Carin (Duke University)</i></p>

      <p><b>MarginGAN: Adversarial Training in Semi-Supervised Learning</b><br><i>Jinhao Dong (Xidian University) &middot; Tong Lin (Peking University)</i></p>

      <p><b>Poincar\&#39;{e} Recurrence, Cycles and Spurious Equilibria in Gradient Descent for Non-Convex Non-Concave Zero-Sum Games</b><br><i>Emmanouil Vlatakis-Gkaragkounis (Columbia University) &middot; Lampros Flokas (Columbia University) &middot; Georgios Piliouras (Singapore University of Technology and Design)</i></p>

      <p><b>A unified variance-reduced accelerated gradient method for convex optimization</b><br><i>Guanghui Lan (Georgia Tech) &middot; Zhize Li (Tsinghua University) &middot; Yi Zhou (IBM Almaden Research Center)</i></p>

      <p><b>Nearly Tight Bounds for Robust Proper Learning of Halfspaces with a Margin</b><br><i>Ilias Diakonikolas (USC) &middot; Daniel Kane (UCSD) &middot; Pasin Manurangsi (Google)</i></p>

      <p><b>Same-Cluster Querying for Overlapping Clusters</b><br><i>Wasim Huleihel (Tel-Aviv University) &middot; Arya Mazumdar (University of Massachusetts Amherst) &middot; Muriel Medard (MIT) &middot; Soumyabrata Pal (University of Massachusetts Amherst)</i></p>

      <p><b>Efficient Convex Relaxations for Streaming PCA</b><br><i>Raman Arora (Johns Hopkins University) &middot; Teodor Vanislavov Marinov (Johns Hopkins University)</i></p>

      <p><b>Learning Robust Global Representations by Penalizing Local Predictive Power</b><br><i>Haohan Wang (Carnegie Mellon University) &middot; Songwei Ge (Carnegie Mellon University) &middot; Zachary Lipton (Carnegie Mellon University) &middot; Eric Xing (Petuum Inc. /  Carnegie Mellon University)</i></p>

      <p><b>Unsupervised Curricula for Visual Meta-Reinforcement Learning</b><br><i>Allan Jabri (UC Berkeley) &middot; Kyle Hsu (University of Toronto) &middot; Ben Eysenbach (Carnegie Mellon University) &middot; Abhishek Gupta (University of California, Berkeley) &middot; Alexei Efros (UC Berkeley) &middot; Sergey Levine (UC Berkeley) &middot; Chelsea Finn (Stanford University)</i></p>

      <p><b>Sample Complexity of Learning Mixture of Sparse Linear Regressions</b><br><i>Akshay Krishnamurthy (Microsoft) &middot; Arya Mazumdar (University of Massachusetts Amherst) &middot; Andrew McGregor (University of Massachusetts Amherst) &middot; Soumyabrata Pal (University of Massachusetts Amherst)</i></p>

      <p><b>Large Scale Adversarial Representation Learning</b><br><i>Jeff Donahue (DeepMind) &middot; Karen Simonyan (DeepMind)</i></p>

      <p><b>G2SAT: Learning to Generate SAT Formulas</b><br><i>Jiaxuan You (Stanford University) &middot; Haoze Wu (Stanford University) &middot; Clark Barrett (Stanford University) &middot; Raghuram Ramanujan (Davidson College) &middot; Jure Leskovec (Stanford University and Pinterest)</i></p>

      <p><b>Neural Proximal Policy Optimization Attains Optimal Policy</b><br><i>Boyi Liu (Northwestern University) &middot; Qi Cai (Northwestern University) &middot; Zhuoran Yang (Princeton University) &middot; Zhaoran Wang (Northwestern University)</i></p>

      <p><b>Dimensionality reduction: theoretical perspective on practical measures</b><br><i>Yair Bartal (Hebrew University) &middot; Nova Fandina (Hebrew University ) &middot; Ofer Neiman (Ben-Gurion University)</i></p>

      <p><b>Oracle-Efficient Algorithms for Online Linear Optimization with Bandit Feedback</b><br><i>Shinji Ito (NEC Corporation,      University of Tokyo) &middot; Daisuke Hatano (RIKEN AIP) &middot; Hanna Sumita (Tokyo Metropolitan University) &middot; Kei Takemura (NEC Corporation) &middot; Takuro Fukunaga (Chuo University, JST PRESTO, RIKEN AIP) &middot; Naonori Kakimura (Keio University) &middot; Ken-Ichi Kawarabayashi (National Institute of Informatics)</i></p>

      <p><b>Multilabel reductions: what is my loss optimising?</b><br><i>Aditya Menon (Google) &middot; Ankit Singh Rawat (Google Research) &middot; Sashank Reddi (Google) &middot; Sanjiv Kumar (Google Research)</i></p>

      <p><b>Tight Sample Complexity of Learning One-hidden-layer Convolutional Neural Networks</b><br><i>Yuan Cao (UCLA) &middot; Quanquan Gu (UCLA)</i></p>

      <p><b>Deep Gamblers: Learning to Abstain with Portfolio Theory</b><br><i>Ziyin Liu (University of Tokyo) &middot; Zhikang Wang (University of Tokyo) &middot; Paul Pu Liang (Carnegie Mellon University) &middot; Ruslan Salakhutdinov (Carnegie Mellon University) &middot; Louis-Philippe Morency (Carnegie Mellon University) &middot; Masahito Ueda (University of Tokyo)</i></p>

      <p><b>Two Time-scale Off-Policy TD Learning: Non-asymptotic Analysis over Markovian Samples</b><br><i>Tengyu Xu (The Ohio State University) &middot; Shaofeng Zou (University at Buffalo, the State University of New York) &middot; Yingbin Liang (The Ohio State University)</i></p>

      <p><b>Transfer Learning via Boosting to Minimize the Performance Gap Between Domains</b><br><i>Boyu Wang (University of Western Ontario) &middot; Jorge A Mendez (University of Pennsylvania) &middot; Mingbo Cai (Princeton University) &middot; Eric Eaton (University of Pennsylvania)</i></p>

      <p><b>Splitting Steepest Descent for Progressive Training of Neural Networks</b><br><i>Lemeng  Wu (UT Austin ) &middot; Dilin Wang (UT Austin) &middot; Qiang Liu (UT Austin)</i></p>

      <p><b>Sequential Experimental Design for Transductive Linear Bandits</b><br><i>Lalit Jain (University of Washington) &middot; Kevin Jamieson (U Washington) &middot; Tanner Fiez (University of Washington) &middot; Lillian Ratliff (University of Washington)</i></p>

      <p><b>Time Matters in Regularizing Deep Networks: Weight Decay and Data Augmentation Affect Early Learning Dynamics, Matter Little Near Convergence</b><br><i>Aditya Sharad Golatkar (UCLA) &middot; Alessandro Achille (UCLA) &middot; Stefano Soatto (UCLA)</i></p>

      <p><b>Outlier-Robust High-Dimensional Sparse Estimation via Iterative Filtering</b><br><i>Ilias Diakonikolas (USC) &middot; Daniel Kane (UCSD) &middot; Sushrut Karmalkar (The University of Texas at Austin) &middot; Eric Price (University of Texas at Austin) &middot; Alistair Stewart (University of Southern California)</i></p>

      <p><b>Variational Graph Recurrent Neural Networks</b><br><i>Ehsan Hajiramezanali (Texas A&M University) &middot; Arman Hasanzadeh (Texas A&M University) &middot; Krishna Narayanan (Texas A&M University) &middot; Nick Duffield (Texas A&M University) &middot; Mingyuan Zhou (University of Texas at Austin) &middot; Xiaoning Qian (Texas A&M)</i></p>

      <p><b>Semi-Implicit Graph Variational Auto-Encoders</b><br><i>Arman Hasanzadeh (Texas A&M University) &middot; Ehsan Hajiramezanali (Texas A&M University) &middot; Krishna Narayanan (Texas A&M University) &middot; Nick Duffield (Texas A&M University) &middot; Mingyuan Zhou (University of Texas at Austin) &middot; Xiaoning Qian (Texas A&M)</i></p>

      <p><b>Unsupervised Learning of Object Keypoints for Perception and Control</b><br><i>Tejas Kulkarni (DeepMind) &middot; Ankush Gupta (DeepMind) &middot; Catalin Ionescu (Deepmind) &middot; Sebastian Borgeaud (DeepMind) &middot; Malcolm Reynolds (DeepMind) &middot; Andrew Zisserman (DeepMind & University of Oxford) &middot; Volodymyr Mnih (DeepMind)</i></p>

      <p><b>InteractiveRecGAN: a Model Based Reinforcement Learning Method with Adversarial Training for Online Recommendation</b><br><i>Xueying Bai (Stony Brook University) &middot; Jian Guan (Tsinghua University) &middot; Hongning Wang (University of Virginia)</i></p>

      <p><b>Optimizing Generalized Rate Metrics through Three-player Games</b><br><i>Harikrishna Narasimhan (Google) &middot; Andrew Cotter (Google) &middot; Maya Gupta (Google)</i></p>

      <p><b>Consistency-based Semi-supervised Learning for Object detection</b><br><i>Jisoo Jeong (Seoul National University) &middot; Seungeui Lee (Seoul National University) &middot; Jeesoo Kim (Seoul National University) &middot; Nojun Kwak (Seoul National University)</i></p>

      <p><b>Rates of Convergence for Large-scale Nearest Neighbor Classification</b><br><i>Xingye Qiao (Binghamton University) &middot; Jiexin Duan (Purdue University) &middot; Guang Cheng (Purdue University)</i></p>

      <p><b>An Embedding Framework for Consistent Polyhedral Surrogates</b><br><i>Jessica Finocchiaro (University of Colorado Boulder) &middot; Rafael Frongillo (CU Boulder) &middot; Bo Waggoner (U. Colorado, Boulder)</i></p>

      <p><b>Cross-Modal Learning with Adversarial Samples</b><br><i>CHAO LI (Xidian University) &middot; Shangqian Gao (University of Pittsburgh) &middot; Cheng Deng (Xidian University) &middot; De Xie (XiDian University) &middot; Wei Liu (Tencent AI Lab)</i></p>

      <p><b>Fast PAC-Bayes via Shifted Rademacher Complexity</b><br><i>Jun Yang (University of Toronto) &middot; Shengyang Sun (University of Toronto) &middot; Daniel Roy (Univ of Toronto & Vector)</i></p>

      <p><b>Cell-Attention Reduces Vanishing Saliency of Recurrent Neural Networks</b><br><i>Aya Abdelsalam Ismail (University of Maryland) &middot; Mohamed Gunady (University of Maryland) &middot; Luiz Pessoa (University of Maryland) &middot; Hector Corrada Bravo (University of Maryland) &middot; Soheil Feizi (University of Maryland, College Park)</i></p>

      <p><b>Program Synthesis and Semantic Parsing with Learned Code Idioms</b><br><i>Richard Shin (UC Berkeley) &middot; Miltiadis Allamanis (Microsoft Research) &middot; Marc Brockschmidt (Microsoft Research) &middot; Alex Polozov (Microsoft Research)</i></p>

      <p><b>Generalization Bounds of Stochastic Gradient Descent for Wide and Deep Neural Networks</b><br><i>Yuan Cao (UCLA) &middot; Quanquan Gu (UCLA)</i></p>

      <p><b>High-Dimensional Optimization in Adaptive Random Subspaces</b><br><i>Jonathan Lacotte (Stanford University) &middot; Mert Pilanci (Stanford) &middot; Marco Pavone (Stanford University)</i></p>

      <p><b>Random Projections with Asymmetric Quantization</b><br><i>Xiaoyun Li (Rutgers University) &middot; Ping Li (Baidu Research USA)</i></p>

      <p><b>Superposition of many models into one</b><br><i>Brian Cheung (UC Berkeley) &middot; Alexander Terekhov (UC Berkeley) &middot; Yubei Chen (UC Berkeley) &middot; Pulkit Agrawal (UC Berkeley) &middot; Bruno Olshausen (Redwood Center/UC Berkeley)</i></p>

      <p><b>Private Testing of Distributions via Sample Permutations</b><br><i>Maryam Aliakbarpour (MIT) &middot; Ilias Diakonikolas (USC) &middot; Daniel Kane (UCSD) &middot; Ronitt Rubinfeld (MIT, TAU)</i></p>

      <p><b>McDiarmid-Type Inequalities for Graph-Dependent Variables and Stability Bounds</b><br><i>Rui (Ray) Zhang (School of Mathematics, Monash University) &middot; Xingwu Liu (University of Chinese Academy of Sciences) &middot; Yuyi Wang (ETH Zurich) &middot; Liwei Wang (Peking University)</i></p>

      <p><b>How to Initialize your Network? Robust Initialization for WeightNorm &amp; ResNets</b><br><i>Devansh Arpit (MILA, UdeM) &middot; Víctor Campos (Barcelona Supercomputing Center) &middot; Yoshua Bengio (U. Montreal)</i></p>

      <p><b>On Making Stochastic Classifiers Deterministic</b><br><i>Andrew Cotter (Google) &middot; Maya Gupta (Google) &middot; Harikrishna Narasimhan (Google)</i></p>

      <p><b>Statistical Analysis of Nearest Neighbor Methods for Anomaly Detection</b><br><i>Xiaoyi Gu (Carnegie Mellon University) &middot; Leman Akoglu (CMU) &middot; Alessandro Rinaldo (CMU)</i></p>

      <p><b>Improving Black-box Adversarial Attacks with a Transfer-based Prior</b><br><i>Shuyu Cheng (Tsinghua University) &middot; Yinpeng Dong (Tsinghua University) &middot; Tianyu Pang (Tsinghua University) &middot; Hang Su (Tsinghua Univiersity) &middot; Jun Zhu (Tsinghua University)</i></p>

      <p><b>Break the Ceiling: Stronger Multi-scale Deep Graph Convolutional Networks</b><br><i>Sitao Luan (McGill University) &middot; Mingde Zhao (Mila, McGill University) &middot; Xiao-Wen Chang (McGill University) &middot; Doina Precup (McGill University / DeepMind Montreal)</i></p>

      <p><b>Statistical Model Aggregation via Parameter Matching</b><br><i>Mikhail Yurochkin (IBM Research, MIT-IBM Watson AI Lab) &middot; Mayank Agarwal (IBM Research) &middot; Soumya Ghosh (IBM Research) &middot; Kristjan Greenewald (IBM Research) &middot; Nghia Hoang (IBM Research)</i></p>

      <p><b>On the (in)fidelity and sensitivity of explanations</b><br><i>Chih-Kuan Yeh (Carnegie Mellon University) &middot; Cheng-Yu Hsieh (National Taiwan University) &middot; Arun Suggala (Carnegie Mellon University) &middot; David Inouye (Carnegie Mellon University) &middot; Pradeep Ravikumar (Carnegie Mellon University)</i></p>

      <p><b>Exponential Family Estimation via Adversarial Dynamics Embedding</b><br><i>Bo Dai (Google Brain) &middot; Zhen Liu (Georgia Institute of Technology) &middot; Hanjun Dai (Georgia Institute of Technology) &middot; Niao He (UIUC) &middot; Arthur Gretton (Gatsby Unit, UCL) &middot; Le Song (Ant Financial & Georgia Institute of Technology) &middot; Dale Schuurmans (Google Inc.)</i></p>

      <p><b>The Broad Optimality of Profile Maximum Likelihood</b><br><i>Yi Hao (University of California, San Diego) &middot; Alon Orlitsky (University of California, San Diego)</i></p>

      <p><b>MintNet: Building Invertible Neural Networks with Masked Convolutions</b><br><i>Yang Song (Stanford University) &middot; Chenlin Meng (Stanford University) &middot; Stefano Ermon (Stanford)</i></p>

      <p><b>Information-Theoretic Generalization Bounds for SGLD via Data-Dependent Estimates</b><br><i>Gintare Karolina Dziugaite (Element AI & University of Cambridge) &middot; Mahdi Haghifam (University of Toronto) &middot; Jeffrey Negrea (University of Toronto) &middot; Ashish  Khisti (University of Toronto) &middot; Daniel Roy (Univ of Toronto & Vector)</i></p>

      <p><b>On Distributed Averaging for Stochastic k-PCA</b><br><i>Aditya Bhaskara (Google Research) &middot; Pruthuvi Wijewardena (University of Utah)</i></p>

      <p><b>Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation</b><br><i>Ke Wang (Peking University) &middot; Hang Hua (Peking University) &middot; Xiaojun Wan (Peking University)</i></p>

      <p><b>MaxGap Bandit: Adaptive Algorithms for Approximate Ranking</b><br><i>Sumeet Katariya (Amazon) &middot; Ardhendu Tripathy (University of Wisconsin - Madison) &middot; Robert Nowak (University of Wisconsion-Madison)</i></p>

      <p><b>Bias Correction of Learned Generative Models using Likelihood-Free Importance Weighting</b><br><i>Aditya Grover (Stanford University) &middot; Jiaming Song (Stanford University) &middot; Ashish Kapoor (Microsoft Research) &middot; Kenneth Tran (Microsoft Research) &middot; Alekh Agarwal (Microsoft Research) &middot; Eric J Horvitz (Microsoft Research) &middot; Stefano Ermon (Stanford)</i></p>

      <p><b>Online Forecasting of Total-Variation-bounded Sequences</b><br><i>Dheeraj Baby () &middot; Yu-Xiang Wang (UC Santa Barbara)</i></p>

      <p><b>Local SGD with  Periodic Averaging: Tighter Analysis  and Adaptive Synchronization </b><br><i>Farzin Haddadpour (Pennsylvania State university) &middot; Mohammad Mahdi Kamani (Pennsylvania State University) &middot; Mehrdad Mahdavi (Pennsylvania State University) &middot; Viveck Cadambe (Penn State)</i></p>

      <p><b>Dynamic Curriculum Learning by Gradient Descent</b><br><i>Shreyas Saxena (Apple Inc.) &middot; Oncel Tuzel (Apple) &middot; Dennis DeCoste (Apple)</i></p>

      <p><b>Unified Sample-Optimal Property Estimation in Near-Linear Time</b><br><i>Yi Hao (University of California, San Diego) &middot; Alon Orlitsky (University of California, San Diego)</i></p>

      <p><b>Region Mutual Information Loss for Semantic Segmentation</b><br><i>Shuai Zhao (Zhejiang University) &middot; Yang Wang (Huazhong University of Science and Technology) &middot; Zheng Yang (FABU) &middot; Deng Cai (ZJU)</i></p>

      <p><b>Learning Stable Deep Dynamics Models</b><br><i>J. Zico Kolter (Carnegie Mellon University / Bosch Center for AI) &middot; Gaurav Manek (Carnegie Mellon University)</i></p>

      <p><b>Image Captioning: Transforming Objects into Words</b><br><i>Simao Herdade (Yahoo Research) &middot; Armin Kappeler (Yahoo Research) &middot; Kofi Boakye (Yahoo Research ) &middot; Joao Soares (Yahoo Research)</i></p>

      <p><b>Greedy Sampling for Approximate Clustering in the Presence of Outliers</b><br><i>Aditya Bhaskara (Google Research) &middot; Sharvaree Vadgama (University of Utah) &middot; Hong Xu (University of Utah)</i></p>

      <p><b>Adversarial Fisher Vectors for Unsupervised Representation Learning</b><br><i>Joshua M Susskind (Apple Inc.) &middot; Shuangfei Zhai (Apple) &middot; Walter Talbott (Apple) &middot; Carlos Guestrin (Apple & University of Washington)</i></p>

      <p><b>On Tractable Computation of Expected Predictions</b><br><i>Pasha Khosravi (UCLA) &middot; YooJung Choi (UCLA) &middot; Yitao Liang (UCLA) &middot; Antonio Vergari (Max-Planck Institute for Intelligent Systems) &middot; Guy Van den Broeck (UCLA)</i></p>

      <p><b>Levenshtein Transformer</b><br><i>Jiatao Gu (Facebook AI Research) &middot; Changhan Wang (Facebook AI Research) &middot; Junbo Zhao (New York University)</i></p>

      <p><b>Unlabeled Data Improves Adversarial Robustness</b><br><i>Yair Carmon (Stanford) &middot; Aditi Raghunathan (Stanford University) &middot; Ludwig Schmidt (UC Berkeley) &middot; John Duchi (Stanford) &middot; Percy Liang (Stanford University)</i></p>

      <p><b>Machine Teaching of Active Sequential Learners</b><br><i>Tomi Peltola (Aalto University) &middot; Mustafa Mert Çelikok (Aalto University) &middot; Pedram Daee (Aalto University) &middot; Samuel Kaski (Aalto University)</i></p>

      <p><b>Gaussian-Based Pooling for Convolutional Neural Networks</b><br><i>Takumi Kobayashi (National Institute of Advanced Industrial Science and Technology)</i></p>

      <p><b>Meta Architecture Search</b><br><i>Albert Shaw (Deepscale) &middot; Wei Wei (Google AI) &middot; Weiyang Liu (Georgia Institute of Technology) &middot; Le Song (Ant Financial & Georgia Institute of Technology) &middot; Bo Dai (Google Brain)</i></p>

      <p><b>NAOMI: Non-Autoregressive Multiresolution Sequence Imputation</b><br><i>Yukai Liu (Caltech) &middot; Rose Yu (Northeastern University) &middot; Stephan Zheng (Salesforce) &middot; Eric Zhan (Caltech) &middot; Yisong Yue (Caltech)</i></p>

      <p><b>Layer-Dependent Importance Sampling for Training Deep and Large Graph Convolutional Networks</b><br><i>Difan Zou (University of California, Los Angeles) &middot; Ziniu Hu (UCLA) &middot; Yewen Wang (UCLA) &middot; Song Jiang (University of California, Los Angeles) &middot; Yizhou Sun (UCLA) &middot; Quanquan Gu (UCLA)</i></p>

      <p><b>Two Generator Game: Learning to Sample via Linear Goodness-of-Fit Test</b><br><i>Lizhong Ding (Inception Institute of Artificial Intelligence) &middot; Mengyang Yu (Inception Institute of Artificial Intelligence) &middot; Li Liu (Inception Institute of Artificial Intelligence) &middot; Fan Zhu (Inception Institute of Artificial Intelligence) &middot; Yong Liu (Institute of Information Engineering, CAS) &middot; Yu Li (King Abdullah University of Science and Technology) &middot; Ling Shao (Inception Institute of Artificial Intelligence)</i></p>

      <p><b>Distribution oblivious, risk-aware algorithms for multi-armed   bandits with unbounded rewards</b><br><i>Anmol Kagrecha (Indian Institute of Technology Bombay) &middot; Jayakrishnan Nair ("Assist. Prof, EE, IIT Bombay") &middot; Krishna Jagannathan (IIT Madras)</i></p>

      <p><b>Private Stochastic Convex Optimization with Optimal Rates</b><br><i>Raef Bassily (The Ohio State University) &middot; Vitaly Feldman (Google Brain) &middot; Kunal Talwar (Google) &middot; Abhradeep Guha Thakurta (University of California Santa Cruz)</i></p>

      <p><b>Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers</b><br><i>Hadi Salman (Microsoft Research AI) &middot; Jerry Li (Microsoft) &middot; Ilya Razenshteyn (Microsoft Research) &middot; Pengchuan Zhang (Microsoft Research) &middot; Huan Zhang (Microsoft Research AI) &middot; Sebastien Bubeck (Microsoft Research) &middot; Greg Yang (Microsoft Research)</i></p>

      <p><b>Demystifying Black-box Models with Symbolic Metamodels</b><br><i>Ahmed Alaa (UCLA) &middot; Mihaela van der Schaar (University of Cambridge, Alan Turing Institute and UCLA)</i></p>

      <p><b>Neural Temporal-Difference Learning Converges to Global Optima</b><br><i>Qi Cai (Northwestern University) &middot; Zhuoran Yang (Princeton University) &middot; Jason Lee (USC) &middot; Zhaoran Wang (Northwestern University)</i></p>

      <p><b>Privacy-Preserving Q-Learning with Functional Noise in Continuous Spaces</b><br><i>Baoxiang Wang (The Chinese University of Hong Kong) &middot; Nidhi Hegde (Borealis AI)</i></p>

      <p><b>Attentive State-Space Modeling of Disease Progression</b><br><i>Ahmed Alaa (UCLA) &middot; Mihaela van der Schaar (University of Cambridge, Alan Turing Institute and UCLA)</i></p>

      <p><b>Online EXP3 Learning in Adversarial Bandits with Delayed Feedback</b><br><i>Ilai Bistritz (Stanford) &middot; Zhengyuan Zhou (Stanford University) &middot; Xi Chen (New York University) &middot; Nicholas Bambos () &middot; Jose Blanchet (Stanford University)</i></p>

      <p><b>A Direct tilde{O}(1/epsilon) Iteration Parallel Algorithm for Optimal Transport</b><br><i>Arun Jambulapati (Stanford University) &middot; Aaron Sidford (Stanford) &middot; Kevin Tian (Stanford University)</i></p>

      <p><b>Faster Boosting with Smaller Memory</b><br><i>Julaiti Alafate (University of California San Diego) &middot; Yoav S Freund (University of California, San Diego)</i></p>

      <p><b>Variance Reduction for Matrix Games</b><br><i>Yair Carmon (Stanford) &middot; Yujia Jin (Stanford University) &middot; Aaron Sidford (Stanford) &middot; Kevin Tian (Stanford University)</i></p>

      <p><b>Learning Neural Networks with Adaptive Regularization</b><br><i>Han Zhao (Carnegie Mellon University) &middot; Yao-Hung Tsai (Carnegie Mellon University) &middot; Ruslan Salakhutdinov (Carnegie Mellon University) &middot; Geoffrey Gordon (MSR Montréal & CMU)</i></p>

      <p><b>Distributed estimation of the inverse Hessian by determinantal averaging</b><br><i>Michal Derezinski (UC Berkeley) &middot; Michael W Mahoney (UC Berkeley)</i></p>

      <p><b>Smoothing Structured Decomposable Circuits</b><br><i>Andy Shih (UCLA) &middot; Guy Van den Broeck (UCLA) &middot; Paul Beame (University of Washington) &middot; Antoine Amarilli (LTCI, Télécom ParisTech)</i></p>

      <p><b>Efficient and Accurate Estimation of Lipschitz Constants for Deep Neural Networks</b><br><i>Mahyar Fazlyab (University of Pennsylvania) &middot; Alexander Robey (University of Pennsylvania) &middot; Hamed Hassani (UPenn) &middot; Manfred Morari (University of Pennsylvania) &middot; George Pappas (University of Pennsylvania)</i></p>

      <p><b>Provable Non-linear Inductive Matrix Completion</b><br><i>Kai Zhong (Amazon) &middot; Zhao Song (UT-Austin) &middot; Prateek Jain (Microsoft Research) &middot; Inderjit S Dhillon (UT Austin & Amazon)</i></p>

      <p><b>Communication-Efficient Distributed Blockwise Momentum SGD with Error-Feedback</b><br><i>Shuai Zheng (HKUST) &middot; Ziyue Huang (Hong Kong University of Science and Technology) &middot; James Kwok (Hong Kong University of Science and Technology)</i></p>

      <p><b>Sparse Variational Inference: Bayesian Coresets from Scratch</b><br><i>Trevor Campbell (UBC) &middot; Boyan Beronov (UBC)</i></p>

      <p><b>Many-Armed Bandits with High-Dimensional Contexts under a Low-Rank Structure</b><br><i>Nima Hamidi (Stanford University) &middot; Mohsen Bayati (Stanford University) &middot; Kapil Gupta (Airbnb)</i></p>

      <p><b>A Necessary and Sufficient Stability Notion for Adaptive Generalization</b><br><i>Moshe Shenfeld (Hebrew University of Jerusalem) &middot; Katrina Ligett (Hebrew University)</i></p>

      <p><b>Necessary and Sufficient Geometries for Adaptive Gradient Algorithms</b><br><i>Daniel Levy (Stanford University) &middot; John Duchi (Stanford)</i></p>

      <p><b>Landmark Ordinal Embedding</b><br><i>Nikhil Ghosh (Caltech) &middot; Yuxin Chen (Caltech) &middot; Yisong Yue (Caltech)</i></p>

      <p><b>Identification of Conditional Causal Effects under Markov Equivalence</b><br><i>Amin Jaber (Purdue University) &middot; Jiji Zhang (Lingnan University) &middot; Elias Bareinboim (Purdue)</i></p>

      <p><b>The Thermodynamic Variational Objective</b><br><i>Vaden Masrani (University of British Columbia) &middot; Tuan Anh Le (University of Oxford) &middot; Frank Wood (University of British Columbia)</i></p>

      <p><b>Global Guarantees for Blind Demodulation with Generative Priors</b><br><i>Paul Hand (Northeastern University) &middot; Babhru Joshi (Rice University)</i></p>

      <p><b>Exact sampling of determinantal point processes with sublinear time preprocessing</b><br><i>Michal Derezinski (UC Berkeley) &middot; Daniele Calandriello (LCSL IIT/MIT) &middot; Michal Valko (DeepMind Paris and Inria Lille - Nord Europe)</i></p>

      <p><b>Geometry-Aware Neural Rendering</b><br><i>Josh Tobin (OpenAI) &middot; Wojciech Zaremba (OpenAI) &middot; Pieter Abbeel (UC Berkeley  Covariant)</i></p>

      <p><b>Variational Temporal Abstraction</b><br><i>Taesup Kim (Mila / Kakao Brain) &middot; Sungjin Ahn (Rutgers University) &middot; Yoshua Bengio (U. Montreal)</i></p>

      <p><b>Subquadratic High-Dimensional Hierarchical Clustering</b><br><i>Amir Abboud (IBM research) &middot; Vincent Cohen-Addad (CNRS & Sorbonne Université) &middot; Hussein Houdrouge (Ecole Polytechnique)</i></p>

      <p><b>Learning Auctions with Robust Incentive Guarantees</b><br><i>Jacob Abernethy (Georgia Institute of Technolog) &middot; Rachel Cummings (Georgia Tech) &middot; Bhuvesh Kumar (Georgia Tech) &middot; Sam Taggart (Oberlin College) &middot; Jamie Morgenstern (Georgia Tech)</i></p>

      <p><b>Policy Optimization Provably Converges to Nash Equilibria in Zero-Sum Linear Quadratic Games</b><br><i>Kaiqing Zhang (University of Illinois at Urbana-Champaign (UIUC)) &middot; Zhuoran Yang (Princeton University) &middot; Tamer Basar ()</i></p>

      <p><b>Uniform convergence may be unable to explain generalization in deep learning</b><br><i>Vaishnavh Nagarajan (Carnegie Mellon University) &middot; J. Zico Kolter (Carnegie Mellon University / Bosch Center for AI)</i></p>

      <p><b>A Zero-Positive Learning Approach for Diagnosing Software Performance Regressions</b><br><i>Mejbah Alam (Intel Labs) &middot; Justin Gottschlich (Intel Labs) &middot; Nesime Tatbul (Intel Labs and MIT) &middot; Javier Turek (Intel Labs) &middot; Timothy Mattson (Intel) &middot; Abdullah Muzahid (Texas A&M University)</i></p>

      <p><b>DTWNet: a Dynamic Time Warping Network</b><br><i>Xingyu Cai (University of Connecticut) &middot; Tingyang Xu (Tencent AI Lab) &middot; Jinfeng Yi (JD Research) &middot; Junzhou Huang (University of Texas at Arlington / Tencent AI Lab) &middot; Sanguthevar Rajasekaran (University of Connecticut)</i></p>

      <p><b>Structured Graph Learning Via Laplacian Spectral Constraints</b><br><i>Sandeep Kumar (Hong Kong University of Science and Technology) &middot;  Jiaxi  Ying (HKUST) &middot; Jose Vinicius de Miranda Cardoso (Universidade Federal de Campina Grande) &middot; Daniel Palomar (The Hong Kong University of Science and Technology)</i></p>

      <p><b>Thresholding Bandit with Optimal Aggregate Regret</b><br><i>Chao Tao (Indiana University Bloomington) &middot; Saúl A Blanco (Indiana University) &middot; Jian Peng (University of Illinois at Urbana-Champaign) &middot; Yuan Zhou (Indiana University Bloomington)</i></p>

      <p><b>Towards Explaining the Regularization Effect of Initial Large Learning Rate in Training Neural Networks</b><br><i>Yuanzhi Li (Princeton) &middot; Colin Wei (Stanford University) &middot; Tengyu Ma (Stanford)</i></p>

      <p><b>Rethinking Kernel Methods for Node Representation Learning on Graphs</b><br><i>Yu Tian (Rutgers) &middot; Long Zhao (Rutgers University) &middot; Xi Peng (University of Delaware) &middot; Dimitris Metaxas (Rutgers University)</i></p>

      <p><b>Causal Misidentification in Imitation Learning</b><br><i>Pim de Haan (University of Amsterdam, visiting at UC Berkeley) &middot; Dinesh Jayaraman (UC Berkeley) &middot; Sergey Levine (UC Berkeley)</i></p>

      <p><b>Optimizing Generalized PageRank Methods for Seed-Expansion Community Detection</b><br><i>Pan Li (Stanford) &middot; I Chien (UIUC) &middot; Olgica Milenkovic (University of Illinois at Urbana-Champaign)</i></p>

      <p><b>The Case for Evaluating Causal Models Using Interventional Measures and Empirical Data</b><br><i>Amanda Gentzel (UMass Amherst) &middot; Dan Garant (C&S Wholesale Grocers) &middot; David Jensen (Univ. of Massachusetts)</i></p>

      <p><b>Dimension-Free Bounds for Low-Precision Training</b><br><i>Zheng Li (Tsinghua University) &middot; Christopher De Sa (Cornell)</i></p>

      <p><b>Concentration of risk measures: A Wasserstein distance approach</b><br><i>Sanjay P. Bhat (Tata Consultancy Services Limited) &middot; Prashanth L.A. (IIT Madras)</i></p>

      <p><b>Meta-Inverse Reinforcement Learning with Probabilistic Context Variables</b><br><i>Lantao Yu (Stanford University) &middot; Tianhe Yu (Stanford University) &middot; Chelsea Finn (Stanford University) &middot; Stefano Ermon (Stanford)</i></p>

      <p><b>Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction</b><br><i>Aviral Kumar (UC Berkeley) &middot; Justin Fu (UC Berkeley) &middot; Matthew Soh (UC Berkeley) &middot; George Tucker (Google Brain) &middot; Sergey Levine (UC Berkeley)</i></p>

      <p><b>Bayesian Optimization with Unknown Search Space</b><br><i>Huong Ha (Deakin University) &middot; Santu Rana (Deakin University) &middot; Sunil Gupta (Deakin University) &middot; Thanh Nguyen (Deakin University) &middot; Hung Tran-The (Deakin University) &middot; Svetha Venkatesh (Deakin University)</i></p>

      <p><b>On the Downstream Performance of Compressed Word Embeddings</b><br><i>Avner May (Stanford University) &middot; Jian Zhang (Stanford University) &middot; Tri Dao (Stanford University) &middot; Christopher Ré (Stanford)</i></p>

      <p><b>Multivariate Distributionally Robust Convex Regression under Absolute Error Loss</b><br><i>Jose Blanchet (Stanford University) &middot; Peter W Glynn (Stanford University) &middot; Jun Yan (Stanford) &middot; Zhengqing Zhou (Stanford University)</i></p>

      <p><b>Neural Relational Inference with Fast Modular Meta-learning</b><br><i>Ferran Alet (MIT) &middot; Erica Weng (MIT) &middot; Tomás Lozano-Pérez (MIT) &middot; Leslie Kaelbling (MIT)</i></p>

      <p><b>Gradient based sample selection for online continual learning </b><br><i>Rahaf Aljundi (KU Leuven, Belgium) &middot; Min Lin (MILA) &middot; Baptiste Goujaud (MILA) &middot; Yoshua Bengio (Mila)</i></p>

      <p><b>Attribution-Based Confidence Metric For Deep Neural Networks</b><br><i>Susmit Jha (SRI International) &middot; Sunny Raj (University of Central Florida) &middot; Steven Fernandes (University of Central Florida) &middot; Sumit Jha (University of Central Florida) &middot; Somesh Jha (University of Wisconsin, Madison) &middot; Brian Jalaian (U.S. Army Research Laboratory) &middot; Gunjan Verma (U.S. Army Research Laboratory) &middot; Ananthram Swami (Army Research Laboratory, Adelphi)</i></p>

      <p><b>Theoretical evidence for adversarial robustness through randomization</b><br><i>Rafael Pinot (Dauphine University - CEA LIST Institute) &middot; Laurent Meunier (Dauphine University - FAIR Paris) &middot; Alexandre Araujo (Université Paris-Dauphine - Wavestone) &middot; Hisashi Kashima (Kyoto University/RIKEN Center for AIP) &middot; Florian Yger (Université Paris-Dauphine) &middot; Cedric Gouy-Pailler (CEA) &middot; Jamal Atif (Université Paris-Dauphine)</i></p>

      <p><b>Online Continual Learning with Maximal Interfered Retrieval</b><br><i>Rahaf Aljundi (KU Leuven, Belgium) &middot; Eugene Belilovsky (University of Montreal) &middot; Tinne Tuytelaars (KU Leuven) &middot; Laurent Charlin (MILA / U.Montreal) &middot; Massimo Caccia (MILA) &middot; Min Lin (MILA) &middot; Lucas Page-Caccia (McGill University)</i></p>

      <p><b>Neural Attribution for Semantic Bug-Localization in Student Programs</b><br><i>Rahul Gupta (Indian Institute of Science) &middot; Aditya Kanade (Indian Institute of Science) &middot; Shirish Shevade (iisc)</i></p>

      <p><b>Adaptive Temporal-Difference Learning for Policy Evaluation with Per-State Uncertainty Estimates</b><br><i>Carlos Riquelme (Google Brain) &middot; Hugo Penedones (Google DeepMind) &middot; Damien Vincent (Google Brain) &middot; Hartmut Maennel (Google) &middot; Sylvain Gelly (Google Brain (Zurich)) &middot; Timothy A Mann (DeepMind) &middot; Andre Barreto (DeepMind) &middot; Gergely Neu (Universitat Pompeu Fabra)</i></p>

      <p><b>SPoC: Search-based Pseudocode to Code</b><br><i>Sumith Kulal (Stanford University) &middot; Panupong Pasupat (Stanford University) &middot; Kartik  Chandra (Stanford University) &middot; Mina Lee (Stanford University) &middot; Oded Padon (Stanford University) &middot; Alex Aiken (Stanford University) &middot; Percy Liang (Stanford University)</i></p>

      <p><b>Generative Modeling by Estimating Gradients of the Data Distribution</b><br><i>Yang Song (Stanford University) &middot; Stefano Ermon (Stanford)</i></p>

      <p><b>Adversarial Music: Real world Audio Adversary against Wake-word Detection System</b><br><i>Juncheng Li (Carnegie Mellon University) &middot; Shuhui Qu (Stanford University) &middot; Xinjian Li (Carnegie Mellon University) &middot; Joseph Szurley (Bosch Center for Artificial Intelligence) &middot; J. Zico Kolter (Carnegie Mellon University / Bosch Center for AI) &middot; Florian Metze (Carnegie Mellon University)</i></p>

      <p><b>Prediction of Spatial Point Processes: Regularized Method with Out-of-Sample Guarantees</b><br><i>Muhammad Osama (Uppsala University) &middot; Dave Zachariah (Uppsala University) &middot; Peter Stoica (Uppsala University)</i></p>

      <p><b>Debiased Bayesian inference for average treatment effects</b><br><i>Kolyan Ray (King's College London) &middot; Botond Szabo (Leiden University)</i></p>

      <p><b>Margin-Based Generalization Lower Bounds for Boosted Classifiers</b><br><i>Allan Grønlund (Aarhus University, MADALGO) &middot; Lior Kamma (Aarhus University) &middot; Kasper Green Larsen (Aarhus University, MADALGO) &middot; Alexander Mathiasen (Aarhus University) &middot; Jelani Nelson ()</i></p>

      <p><b>Connections Between Mirror Descent, Thompson Sampling and the Information Ratio</b><br><i>Julian Zimmert (University of Copenhagen) &middot; Tor Lattimore (DeepMind)</i></p>

      <p><b>Graph Transformer Networks</b><br><i>Seongjun Yun (Korea university) &middot; Minbyul Jeong (Korea university) &middot; Raehyun Kim (Korea university) &middot; Jaewoo Kang (Korea University) &middot; Hyunwoo Kim (Korea University)</i></p>

      <p><b>Learning to Confuse: Generating Training Time Adversarial Data with Auto-Encoder</b><br><i>Ji Feng (Sinovation Ventures) &middot; Qi-Zhi Cai (Sinovation Ventures) &middot; Zhi-Hua Zhou (Nanjing University)</i></p>

      <p><b>The Impact of Regularization on High-dimensional Logistic Regression</b><br><i>Fariborz Salehi (California Institute of Technology) &middot; Ehsan Abbasi (Caltech) &middot; Babak Hassibi (Caltech)</i></p>

      <p><b>Adaptive Density Estimation for Generative Models</b><br><i>Thomas LUCAS (Inria Grenoble) &middot; Konstantin Shmelkov (Huawei) &middot; Karteek Alahari (Inria) &middot; Cordelia Schmid (Inria / Google) &middot; Jakob Verbeek (INRIA)</i></p>

      <p><b>Fast and Provable ADMM for Learning with Generative Priors</b><br><i>Fabian Latorre Gomez (EPFL) &middot; Armin eftekhari (EPFL) &middot; Volkan Cevher (EPFL)</i></p>

      <p><b>Weighted Linear Bandits for Non-Stationary Environments</b><br><i>Yoan Russac (Ecole Normale Supérieure) &middot; Claire Vernade (Google DeepMind) &middot; Olivier Cappé (CNRS)</i></p>

      <p><b>Improved Regret Bounds for Bandit Combinatorial Optimization</b><br><i>Shinji Ito (NEC Corporation,      University of Tokyo) &middot; Daisuke Hatano (RIKEN AIP) &middot; Hanna Sumita (Tokyo Metropolitan University) &middot; Kei Takemura (NEC Corporation) &middot; Takuro Fukunaga (Chuo University, JST PRESTO, RIKEN AIP) &middot; Naonori Kakimura (Keio University) &middot; Ken-Ichi Kawarabayashi (National Institute of Informatics)</i></p>

      <p><b>Pareto Multi-Task Learning</b><br><i>Xi Lin (City University of Hong Kong) &middot; Hui-Ling Zhen (City University of Hong Kong) &middot; Zhenhua Li (National University of Singapore) &middot; Qing-Fu Zhang () &middot; Sam Kwong (City Univeristy of Hong Kong)</i></p>

      <p><b>SIC-MMAB: Synchronisation Involves Communication in Multiplayer Multi-Armed Bandits</b><br><i>Etienne Boursier (ENS Paris Saclay) &middot; Vianney Perchet (ENS Paris-Saclay & Criteo AI Lab)</i></p>

      <p><b>Novel positional encodings to enable tree-based transformers</b><br><i>Vighnesh Shiv (Microsoft Research) &middot; Chris Quirk (Microsoft Research)</i></p>

      <p><b>A Domain Agnostic Measure for Monitoring and Evaluating GANs</b><br><i>Paulina Grnarova (ETH Zurich) &middot; Yehuda Kfir Levy (ETH) &middot; Aurelien Lucchi (ETH Zurich) &middot; Nathanael Perraudin (Swiss Data Science Center - EPFL / ETH Zurich) &middot; Ian Goodfellow (Google) &middot; Thomas Hofmann (ETH Zurich) &middot; Andreas Krause (ETH Zurich)</i></p>

      <p><b>Submodular Function Minimization with Noisy Evaluation Oracle</b><br><i>Shinji Ito (NEC Corporation,      University of Tokyo)</i></p>

      <p><b>Counting the Optimal Solutions in Graphical Models</b><br><i>Radu Marinescu (IBM Research) &middot; Rina Dechter (UCI)</i></p>

      <p><b>Modelling the Dynamics of Multiagent Q-Learning in Repeated Symmetric Games: a Mean Field Theoretic Approach</b><br><i>Shuyue Hu (the Chinese University of Hong Kong) &middot; Chin-wing Leung (The Chinese University of Hong Kong) &middot; Ho-fung Leung (The Chinese University of Hong Kong)</i></p>

      <p><b>Deep Multimodal Multilinear Fusion with High-order Polynomial Pooling</b><br><i>Ming Hou (RIKEN AIP) &middot; Jiajia Tang (Hangzhou Dianzi University / RIKEN AIP) &middot; Jianhai Zhang (Hangzhou Dianzi University) &middot; Wanzeng Kong (Hangzhou Dianzi University) &middot; Qibin Zhao (RIKEN AIP)</i></p>

      <p><b>Bootstrapping Upper Confidence Bound</b><br><i>Botao Hao (Purdue University) &middot; Yasin Abbasi (Adobe Research) &middot; Zheng Wen (Adobe Research) &middot; Guang Cheng (Purdue University)</i></p>

      <p><b>Integer Discrete Flows and Lossless Compression</b><br><i>Emiel Hoogeboom (University of Amsterdam) &middot; Jorn Peters (University of Amsterdam) &middot; Rianne van den Berg (Google Brain) &middot; Max Welling (University of Amsterdam / Qualcomm AI Research)</i></p>

      <p><b>Structured Prediction with Projection Oracles</b><br><i>Mathieu Blondel (NTT)</i></p>

      <p><b>Primal Dual Formulation For Deep Learning With Constraints</b><br><i>Yatin Nandwani (Indian Institute Of Technology Delhi) &middot; Abhishek Pathak (Indian Institute Of Technology, Delhi) &middot; Mausam  (IIT Dehli) &middot; Parag Singla (Indian Institute of Technology Delhi)</i></p>

      <p><b>Screening Sinkhorn Algorithm for Regularized Optimal Transport</b><br><i>Mokthar Z. Alaya (University of Rouen) &middot; Maxime Berar (Université de Rouen) &middot; Gilles Gasso (LITIS - INSA de Rouen) &middot; Alain Rakotomamonjy (Université de Rouen Normandie   Criteo AI Lab)</i></p>

      <p><b>PAC-Bayes Un-Expected Bernstein Inequality</b><br><i>Zakaria Mhammedi (The Australian National University) &middot; Peter Grünwald (CWI and Leiden University) &middot; Benjamin Guedj (Inria & University College London)</i></p>

      <p><b>Are Labels Required for Improving Adversarial Robustness?</b><br><i>Jean-Baptiste Alayrac (Deepmind) &middot; Jonathan Uesato (DeepMind) &middot; Po-Sen Huang (DeepMind) &middot; Alhussein Fawzi (DeepMind) &middot; Robert Stanforth (DeepMind) &middot; Pushmeet Kohli (DeepMind)</i></p>

      <p><b>Tight Regret Bounds for Model-Based Reinforcement Learning with Greedy Policies</b><br><i>Yonathan Efroni (Technion) &middot; Nadav Merlis (Technion) &middot; Mohammad Ghavamzadeh (Facebook AI Research) &middot; Shie Mannor (Technion)</i></p>

      <p><b>Multi-objective Bayesian optimisation with preferences over objectives</b><br><i>Majid Abdolshah (Deakin University) &middot; Alistair Shilton (Deakin University) &middot; Santu Rana (Deakin University) &middot; Sunil Gupta (Deakin University) &middot; Svetha Venkatesh (Deakin University)</i></p>

      <p><b>Think out of the &quot;Box&quot;: Generically-Constrained Asynchronous Composite Optimization and Hedging</b><br><i>Pooria Joulani (DeepMind) &middot; András György (DeepMind) &middot; Csaba Szepesvari (DeepMind/University of Alberta)</i></p>

      <p><b>Calibration tests in multi-class classification: A unifying framework</b><br><i>David Widmann (Uppsala University) &middot; Fredrik Lindsten (Linköping Universituy) &middot; Dave Zachariah (Uppsala University)</i></p>

      <p><b>Classification Accuracy Score for Conditional Generative Models</b><br><i>Suman Ravuri (DeepMind) &middot; Oriol Vinyals (Google DeepMind)</i></p>

      <p><b>Theoretical Analysis Of Adversarial Learning: A Minimax Approach</b><br><i>Zhuozhuo Tu (The University of Sydney) &middot; Jingwei Zhang (Hong Kong University of Science and Technology & University of Sydney) &middot; Dacheng Tao (University of Sydney)</i></p>

      <p><b>Multiagent Evaluation under Incomplete Information</b><br><i>Mark Rowland (DeepMind) &middot; Shayegan Omidshafiei (DeepMind) &middot; Karl Tuyls (DeepMind) &middot; Julien Perolat (DeepMind) &middot; Michal Valko (DeepMind Paris and Inria Lille - Nord Europe) &middot; Georgios Piliouras (Singapore University of Technology and Design) &middot; Remi Munos (DeepMind)</i></p>

      <p><b>Tree-Sliced Variants of Wasserstein Distances</b><br><i>Tam Le (RIKEN AIP) &middot; Makoto Yamada (Kyoto University / RIKEN AIP) &middot; Kenji Fukumizu (Institute of Statistical Mathematics / Preferred Networks / RIKEN AIP) &middot; Marco Cuturi (Google and CREST/ENSAE)</i></p>

      <p><b>Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration</b><br><i>Meelis Kull (University of Tartu) &middot; Miquel Perello Nieto (University of Bristol) &middot; Markus Kängsepp (University of Tartu) &middot; Telmo Silva Filho (Universidade Federal da Paraíba) &middot; Hao Song (University of Bristol) &middot; Peter Flach (University of Bristol)</i></p>

      <p><b>Comparing distributions: $\ell_1$ geometry improves kernel two-sample testing</b><br><i>meyer scetbon (ENS CACHAN) &middot; Gael Varoquaux (Parietal Team, INRIA)</i></p>

      <p><b>Robustness Verification of Tree-based Models</b><br><i>Hongge Chen (MIT) &middot; Huan Zhang (UCLA) &middot; Si Si (Google Research) &middot; Yang Li (Google) &middot; Duane Boning (Massachusetts Institute of Technology) &middot; Cho-Jui Hsieh (UCLA)</i></p>

      <p><b>Towards Interpretable Reinforcement Learning Using Attention Augmented Agents</b><br><i>Alexander Mott (DeepMind) &middot; Daniel Zoran (DeepMind) &middot; Mike Chrzanowski (DeepMind) &middot; Daan Wierstra (DeepMind Technologies) &middot; Danilo Jimenez Rezende (Google DeepMind)</i></p>

      <p><b>Fast and Accurate Stochastic Gradient Estimation</b><br><i>Beidi Chen (Rice University) &middot; Yingchen Xu (Rice University) &middot; Anshumali Shrivastava (Rice University)</i></p>

      <p><b>Theoretical Limits of Pipeline Parallel Optimization and Application to Distributed Deep Learning</b><br><i>Igor Colin (Huawei) &middot; Ludovic DOS SANTOS (Huawei) &middot; Kevin Scaman (Huawei Technologies, Noah's Ark)</i></p>

      <p><b>Root Mean Square Layer Normalization</b><br><i>Biao Zhang (University of Edinburgh) &middot; Rico Sennrich (University of Edinburgh)</i></p>

      <p><b>Universality in Learning from Linear Measurements</b><br><i>Ehsan Abbasi (Caltech) &middot; Fariborz Salehi (California Institute of Technology) &middot; Babak Hassibi (Caltech)</i></p>

      <p><b>Planning in Entropy-Regularized Markov Decision Processes and Games</b><br><i>Jean-Bastien Grill (Google DeepMind) &middot; Omar Darwiche Domingues (Inria) &middot; Pierre Menard (Inria) &middot; Remi Munos (DeepMind) &middot; Michal Valko (DeepMind Paris and Inria Lille - Nord Europe)</i></p>

      <p><b>Exponentially convergent stochastic k-PCA without variance reduction</b><br><i>Cheng Tang (Amazon)</i></p>

      <p><b>R2D2: Reliable and Repeatable Detectors and Descriptors for Joint Sparse Keypoint Detection and Local Feature Extraction</b><br><i>Jerome Revaud (Naver Labs Europe) &middot; Cesar De Souza (NAVER LABS Europe) &middot; Martin Humenberger (Naver Labs Europe) &middot; Philippe Weinzaepfel (NAVER LABS Europe)</i></p>

      <p><b>Selective Sampling-based Scalable Sparse Subspace Clustering</b><br><i>Shin Matsushima (The University of Tokyo) &middot; Maria Brbic (Stanford University)</i></p>

      <p><b>A General Framework for Efficient Symmetric Property Estimation</b><br><i>Moses Charikar (Stanford University) &middot; Kirankumar Shiragur (Stanford University) &middot; Aaron Sidford (Stanford)</i></p>

      <p><b>Structured Variational Inference in Continuous Cox Process Models</b><br><i>Virginia Aglietti (University of Warwick) &middot; Edwin Bonilla (CSIRO's Data61) &middot; Theodoros Damoulas (University of Warwick        The Alan Turing Institute) &middot; Sally  Cripps (University of Sydney)</i></p>

      <p><b>Generalization of Reinforcement Learners with Working and Episodic Memory</b><br><i>Meire Fortunato (DeepMind) &middot; Melissa Tan (Deepmind) &middot; Ryan Faulkner (Deepmind) &middot; Steven Hansen (DeepMind) &middot; Adrià Puigdomènech Badia (Google DeepMind) &middot; Gavin Buttimore (DeepMind) &middot; Charles Deck (Deepmind) &middot; Joel Leibo (DeepMind) &middot; Charles Blundell (DeepMind)</i></p>

      <p><b>Distribution Learning of a Random Spatial Field with a Location-Unaware Mobile Sensor</b><br><i>Meera V Pai (Indian Institute of Technology Bombay) &middot; Animesh Kumar (Indian Institute of Technology Bombay)</i></p>

      <p><b>Hindsight Credit Assignment</b><br><i>Anna Harutyunyan (DeepMind) &middot; Will Dabney (DeepMind) &middot; Thomas Mesnard (DeepMind) &middot; Mohammad Gheshlaghi Azar (DeepMind) &middot; Bilal Piot (DeepMind) &middot; Nicolas Heess (Google DeepMind) &middot; Hado van Hasselt (DeepMind) &middot; Gregory Wayne (Google DeepMind) &middot; Satinder Singh (DeepMind) &middot; Doina Precup (DeepMind) &middot; Remi Munos (DeepMind)</i></p>

      <p><b>Efficient Identification in Linear Structural Causal Models with Instrumental Cutsets</b><br><i>Daniel Kumor (Purdue) &middot; Bryant Chen (Brex) &middot; Elias Bareinboim (Purdue)</i></p>

      <p><b>Kernelized Bayesian Softmax for Text Generation</b><br><i>NING MIAO (Peking University) &middot; Hao Zhou (Bytedance) &middot; Chengqi Zhao (Bytedance) &middot; Wenxian Shi (Bytedance) &middot; Yitan Li (ByteDance.Inc) &middot; Lei Li (Bytedance)</i></p>

      <p><b>When to Trust Your Model: Model-Based Policy Optimization</b><br><i>Michael Janner (UC Berkeley) &middot; Justin Fu (UC Berkeley) &middot; Marvin Zhang (UC Berkeley) &middot; Sergey Levine (UC Berkeley)</i></p>

      <p><b>Correlation Clustering with Adaptive Similarity Queries</b><br><i>Marco Bressan (Sapienza University of Rome) &middot; Nicolò Cesa-Bianchi (Università degli Studi di Milano) &middot; Andrea Paudice (University of Milan) &middot; Fabio Vitale (Sapienza University of Rome)</i></p>

      <p><b>Control What You Can: Intrinsically Motivated Task-Planning Agent</b><br><i>Sebastian Blaes (Max Planck Institute for Intelligent Systems) &middot; Marin Vlastelica Pogančić (Max-Planck Institute for Intelligent Systems, Tuebingen) &middot; Jia-Jie Zhu (Max Planck Institute for Intelligent Systems) &middot; Georg Martius (MPI for Intelligent Systems)</i></p>

      <p><b>Selecting causal brain features with a single conditional independence test per feature</b><br><i>Atalanti Mastakouri (Max Planck Institute for Intelligent Systems) &middot; Bernhard Schölkopf (MPI for Intelligent Systems) &middot; Dominik Janzing (Amazon)</i></p>

      <p><b>Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders </b><br><i>Emile Mathieu () &middot; Charline Le Lan (University of Oxford) &middot; Chris J. Maddison (Institute for Advanced Study, Princeton) &middot; Ryota Tomioka (Microsoft Research Cambridge) &middot; Yee Whye Teh (University of Oxford, DeepMind)</i></p>

      <p><b>A Generic Acceleration Framework for Stochastic Composite Optimization</b><br><i>Andrei Kulunchakov (Inria) &middot; Julien Mairal (Inria)</i></p>

      <p><b>Beating SGD Saturation with Tail-Averaging  and Minibatching</b><br><i>Nicole Muecke (University of Stuttgart) &middot; Gergely Neu (Universitat Pompeu Fabra) &middot; Lorenzo Rosasco (University of Genova- MIT - IIT)</i></p>

      <p><b>Random Quadratic Forms with Dependence: Applications to Restricted Isometry and Beyond</b><br><i>Arindam Banerjee (Voleon) &middot; Qilong Gu (University of Minnesota Twin Cities) &middot; Vidyashankar Sivakumar (University of Minnesota) &middot; Steven Wu (Microsoft Research)</i></p>

      <p><b>Continuous-time Models for Stochastic Optimization Algorithms</b><br><i>Antonio Orvieto (ETH Zurich) &middot; Aurelien Lucchi (ETH Zurich)</i></p>

      <p><b>Curriculum-guided Hindsight Experience Replay</b><br><i>Meng Fang (Tencent) &middot; Tianyi Zhou (University of Washington, Seattle) &middot; Yali Du (University of Technology Sydney) &middot; Lei Han (Rutgers University) &middot; Zhengyou Zhang ()</i></p>

      <p><b>Implicit Semantic Data Augmentation for Deep Networks</b><br><i>Yulin Wang (Tsinghua University) &middot; Xuran Pan (Tsinghua University) &middot; Shiji Song (Department of Automation, Tsinghua University) &middot; Hong Zhang (Baidu Inc.) &middot; Gao Huang (Tsinghua) &middot; Cheng Wu (Tsinghua)</i></p>

      <p><b>MetaInit: Initializing learning by learning to initialize</b><br><i>Yann Dauphin (Google AI) &middot; Samuel Schoenholz (Google Brain)</i></p>

      <p><b>Scalable Deep Generative Relational Model with High-Order Node Dependence</b><br><i>Xuhui Fan (University of New South Wales) &middot; Bin Li (Fudan University) &middot; Caoyuan Li (UTS) &middot; Scott SIsson (University of New South Wales, Sydney) &middot; Ling Chen (" University of Technology, Sydney, Australia")</i></p>

      <p><b>Random Path Selection for Continual Learning</b><br><i>Jathushan Rajasegaran (IIAI) &middot; Munawar Hayat (IIAI) &middot; Salman  Khan (IIAI) &middot; Fahad Shahbaz Khan (Inception Institute of Artificial Intelligence) &middot; Ling Shao (Inception Institute of Artificial Intelligence)</i></p>

      <p><b>Efficient Algorithms for Smooth Minimax Optimization</b><br><i>Kiran Thekumparampil (Univ. of Illinois at Urbana-Champaign) &middot; Prateek Jain (Microsoft Research) &middot; Praneeth Netrapalli (Microsoft Research) &middot; Sewoong Oh (University of Washington)</i></p>

      <p><b>Shadowing Properties of Optimization Algorithms</b><br><i>Antonio Orvieto (ETH Zurich) &middot; Aurelien Lucchi (ETH Zurich)</i></p>

      <p><b>Causal Regularization</b><br><i>Dominik Janzing (Amazon)</i></p>

      <p><b>Learning Hawkes Processes from a handful of events</b><br><i>Farnood Salehi (EPFL) &middot; William Trouleau (EPFL) &middot; Matthias Grossglauser (EPFL) &middot; Patrick Thiran (EPFL)</i></p>

      <p><b>Unsupervised Object Segmentation by Redrawing</b><br><i>Mickael Chen (Université Pierre et Marie Curie) &middot; Thierry Artières (Aix-Marseille Université) &middot; Ludovic Denoyer (Facebook - FAIR)</i></p>

      <p><b>Regret Bounds for Learning State Representations in Reinforcement Learning</b><br><i>Ronald Ortner (Montanuniversitaet Leoben) &middot; Matteo Pirotta (Facebook AI Research) &middot; Alessandro Lazaric (Facebook Artificial Intelligence Research) &middot; Ronan Fruit (Inria Lille) &middot; Odalric-Ambrym Maillard (INRIA)</i></p>

      <p><b>Band-Limited Gaussian Processes: The Sinc Kernel</b><br><i>Felipe Tobar (Universidad de Chile)</i></p>

      <p><b>Leveraging Labeled and Unlabeled Data for Consistent Fair Binary Classification</b><br><i>Evgenii Chzhen (Université Paris-Est) &middot; Christophe Denis (Universit? Paris Est) &middot; Mohamed Hebiri () &middot; Luca Oneto (University of Genoa) &middot; Massimiliano Pontil (IIT)</i></p>

      <p><b>Learning search spaces for Bayesian optimization: Another view of hyperparameter transfer learning</b><br><i>Valerio Perrone (Amazon) &middot; Huibin Shen (Amazon) &middot; Matthias Seeger (Amazon) &middot; Cedric Archambeau (Amazon) &middot; Rodolphe Jenatton (Amazon)</i></p>

      <p><b>Feedforward Bayesian Inference for Crowdsourced Classification</b><br><i>Edoardo Manino (University of Southampton) &middot; Long Tran-Thanh (University of Southampton) &middot; Nicholas Jennings (Imperial College, London)</i></p>

      <p><b>Neuropathic Pain Diagnosis Simulator for Causal Discovery Algorithm Evaluation</b><br><i>Ruibo Tu (KTH Royal Institute of Technology) &middot; Kun Zhang (CMU) &middot; Bo Bertilson (KI Karolinska Institutet) &middot; Hedvig Kjellstrom (KTH Royal Institute of Technology) &middot; Cheng Zhang (Microsoft)</i></p>

      <p><b>Brain-Like Object Recognition with High-Performing Shallow Recurrent ANNs</b><br><i>Jonas Kubilius (Massachusetts Institute of Technology) &middot; Martin Schrimpf (MIT) &middot; Ha Hong (Bay Labs Inc.) &middot; Najib Majaj (NYU) &middot; Rishi Rajalingham (MIT) &middot; Elias Issa (Columbia University) &middot; Kohitij Kar (MIT) &middot; Pouya Bashivan (Massachusetts Institute of Technology) &middot; Jonathan Prescott-Roy (MIT) &middot; Kailyn Schmidt (MIT) &middot; Aran Nayebi (Stanford University) &middot; Daniel Bear (Stanford University) &middot; Daniel Yamins (Stanford University) &middot; James J DiCarlo (Massachusetts Institute of Technology)</i></p>

      <p><b>k-Means Clustering of Lines for Big Data</b><br><i>Yair Marom (University of Haifa) &middot; Dan Feldman (University of Haifa)</i></p>

      <p><b>Random projections and sampling algorithms for clustering of high-dimensional polygonal curves</b><br><i>Stefan Meintrup (TU Dortmund) &middot; Alexander Munteanu (TU Dortmund) &middot; Dennis Rohde (TU Dortmund)</i></p>

      <p><b>Recurrent Space-time Graph Neural Networks</b><br><i>Andrei  Nicolicioiu (Bitdefender) &middot; Iulia Duta (Bitdefender) &middot; Marius Leordeanu (Institute of Mathematics of the Romanian Academy)</i></p>

      <p><b>Uncertainty on Asynchronous Event Prediction</b><br><i>Bertrand Charpentier (Technical University of Munich) &middot; Marin Biloš (Technical University of Munich) &middot; Stephan Günnemann (Technical University of Munich)</i></p>

      <p><b>Accurate, reliable and fast robustness evaluation</b><br><i>Wieland Brendel (AG Bethge, University of Tübingen) &middot; Jonas Rauber (University of Tübingen) &middot; Matthias Kümmerer (University of Tübingen) &middot; Ivan Ustyuzhaninov (University of Tübingen) &middot; Matthias Bethge (University of Tübingen)</i></p>

      <p><b>Sparse High-Dimensional Isotonic Regression</b><br><i>David Gamarnik (Massachusetts Institute of Technology) &middot; Julia Gaudio (Massachusetts Institute of Technology)</i></p>

      <p><b>Triad Constraints for Learning Causal Structure of Latent Variables</b><br><i>Ruichu Cai (Guangdong University of Technology) &middot; Feng Xie (Guangdong University of Technology) &middot; Clark Glymour (Carnegie Mellon University) &middot; Zhifeng Hao (Guangdong University of Technology) &middot; Kun Zhang (CMU)</i></p>

      <p><b>On the Inductive Bias of Neural Tangent Kernels</b><br><i>Alberto Bietti (Inria) &middot; Julien Mairal (Inria)</i></p>

      <p><b>Cross-Domain Transferable Perturbations</b><br><i>Muzammal Naseer (Australian National University (ANU)) &middot; Salman  Khan (IIAI) &middot; Muhammad Haris Khan (Inception Institute of Artificial Intelligence) &middot; Fahad Shahbaz Khan (Inception Institute of Artificial Intelligence) &middot; Fatih Porikli (ANU)</i></p>

      <p><b>Shallow RNN:  Accurate Time-series Classification on Resource Constrained Devices</b><br><i>Don Dennis (Microsoft Research) &middot; Durmus Alp Emre Acar (Boston University) &middot; Vikram Mandikal (Microsoft Research) &middot; Vinu Sankar Sadasivan (Indian Institute of Technology Gandhinagar) &middot; Venkatesh Saligrama (Boston University) &middot; Harsha Vardhan Simhadri (Microsoft Research India) &middot; Prateek Jain (Microsoft Research)</i></p>

      <p><b>Kernel quadrature with DPPs</b><br><i>Ayoub Belhadji (Ecole Centrale de Lille) &middot; Rémi Bardenet (University of Lille) &middot; Pierre Chainais (Centrale Lille / CRIStAL CNRS UMR 9189)</i></p>

      <p><b>REM: From Structural Entropy to Community Structure Deception </b><br><i>Yiwei Liu (Beijing institute of technology) &middot; Jiamou Liu (University of Auckland) &middot; Zijian Zhang (Beijing Institute of Technology) &middot; Liehuang Zhu (Beijing Institute of Technology) &middot; Angsheng Li (Beihang University)</i></p>

      <p><b>Sim2real transfer learning for 3D pose estimation: motion to the rescue</b><br><i>Carl Doersch (DeepMind) &middot; Andrew Zisserman (DeepMind & University of Oxford)</i></p>

      <p><b>Self-Supervised Deep Learning on Point Clouds by Reconstructing Space</b><br><i>Bjarne Sievers (Hasso-Plattner-Institut) &middot; Jonathan Sauder (Hasso Plattner Institute)</i></p>

      <p><b>Piecewise Strong Convexity of Neural Networks</b><br><i>Tristan Milne (University of Toronto)</i></p>

      <p><b>Minimum Stein Discrepancy Estimators</b><br><i>Alessandro Barp (Imperial College London) &middot; Francois-Xavier Briol (University of Cambridge) &middot; Andrew Duncan (Imperial College London) &middot; Mark Girolami (University of Cambridge) &middot; Lester Mackey (Microsoft Research)</i></p>

      <p><b>Fast and Furious Learning in Zero-Sum Games: Vanishing Regret with Non-Vanishing Step Sizes</b><br><i>James Bailey (Singapore University of Technology and Design) &middot; Georgios Piliouras (Singapore University of Technology and Design)</i></p>

      <p><b>Generalization Bounds for Neural Networks via Approximate Description Length</b><br><i>Amit Daniely (Google Research) &middot; Elad Granot (Hebrew University)</i></p>

      <p><b>Provably robust boosted decision stumps and trees against adversarial attacks</b><br><i>Maksym Andriushchenko (University of Tübingen / EPFL) &middot; Matthias Hein (University of Tübingen)</i></p>

      <p><b>Convergence of Adversarial Training in Overparametrized Neural Networks</b><br><i>Ruiqi Gao (Peking University) &middot; Tianle Cai (Peking University) &middot; Haochuan Li (MIT) &middot; Cho-Jui Hsieh (UCLA) &middot; Liwei Wang (Peking University) &middot; Jason Lee (USC)</i></p>

      <p><b>A Composable Specification Language for Reinforcement Learning Tasks</b><br><i>Kishor Jothimurugan (University of Pennsylvania) &middot; Rajeev Alur (University of Pennsylvania ) &middot; Osbert Bastani (University of Pennysylvania)</i></p>

      <p><b>The Option Keyboard: Combining Skills in Reinforcement Learning</b><br><i>Andre Barreto (DeepMind) &middot; Diana Borsa (DeepMind) &middot; Shaobo Hou (DeepMind) &middot; Gheorghe Comanici (Google) &middot; Eser Aygun (Google Canada) &middot; Philippe Hamel (Google) &middot; Daniel Toyama (DeepMind Montreal) &middot; Jonathan J Hunt (DeepMind) &middot; Shibl Mourad (Google) &middot; David Silver (DeepMind) &middot; Doina Precup (DeepMind)</i></p>

      <p><b>Unified Language Model Pre-training for Natural Language Understanding and Generation</b><br><i>Li Dong (Microsoft Research) &middot; Nan Yang (Microsoft Research Asia) &middot; Wenhui Wang (Microsoft Research) &middot; Furu Wei (Microsoft Research Asia) &middot; Xiaodong Liu (Microsoft) &middot; Yu Wang (Microsoft Research) &middot; Jianfeng Gao (Microsoft Research, Redmond, WA) &middot; Ming Zhou (Microsoft Research) &middot; Hsiao-Wuen Hon (Microsoft Research)</i></p>

      <p><b>Learning to Correlate in Multi-Player General-Sum Sequential Games</b><br><i>Andrea Celli (Politecnico di Milano) &middot; Alberto Marchesi (Politecnico di Milano) &middot; Tommaso Bianchi (Politecnico di Milano) &middot; Nicola Gatti (Politecnico di Milano)</i></p>

      <p><b>Stochastic Continuous Greedy ++:  When Upper and Lower Bounds Match</b><br><i>Amin Karbasi (Yale) &middot; Hamed Hassani (UPenn) &middot; Aryan Mokhtari (UT Austin) &middot; Zebang Shen (Zhejiang University)</i></p>

      <p><b>Generative Well-intentioned Networks</b><br><i>Justin T Cosentino (Tsinghua University) &middot; Jun Zhu (Tsinghua University)</i></p>

      <p><b>Online-Within-Online Meta-Learning</b><br><i>Giulia Denevi (IIT/UNIGE) &middot; Dimitris Stamos (University College London) &middot; Carlo Ciliberto (Imperial College London) &middot; Massimiliano Pontil (IIT & UCL)</i></p>

      <p><b>Learning step sizes for unfolded sparse coding</b><br><i>Pierre Ablin (Inria) &middot; Thomas Moreau (Inria) &middot; Mathurin Massias (Inria) &middot; Alexandre Gramfort (INRIA, Université Paris-Saclay)</i></p>

      <p><b>Biases for Emergent Communication in Multi-agent Reinforcement Learning</b><br><i>Tom Eccles (DeepMind) &middot; Yoram Bachrach () &middot; Guy Lever (Google DeepMind) &middot; Angeliki Lazaridou (DeepMind) &middot; Thore Graepel (DeepMind)</i></p>

      <p><b>Episodic Memory in Lifelong Language Learning</b><br><i>Cyprien de Masson d'Autume (Google DeepMind) &middot; Sebastian Ruder (DeepMind) &middot; Lingpeng Kong (DeepMind) &middot; Dani Yogatama (DeepMind)</i></p>

      <p><b>A Simple Baseline for Bayesian Uncertainty in Deep Learning</b><br><i>Wesley J Maddox (Cornell University) &middot; Pavel Izmailov (CORNELL UNIVERSITY) &middot; Timur Garipov (MIT) &middot; Dmitry Vetrov (Higher School of Economics, Samsung AI Center, Moscow) &middot; Andrew Wilson (Cornell University)</i></p>

      <p><b>Communication-efficient Distributed SGD with Sketching</b><br><i>Nikita Ivkin (Amazon) &middot; Daniel Rothchild (UC Berkeley) &middot; Md Enayat Ullah (Johns Hopkins University) &middot; Vladimir braverman (Johns Hopkins University) &middot; Ion Stoica (UC Berkeley) &middot; Raman Arora (Johns Hopkins University)</i></p>

      <p><b>Modeling Conceptual Understanding in Image Reference Games</b><br><i>Rodolfo Corona Rodriguez (University of Amsterdam) &middot; Zeynep Akata (University of Amsterdam) &middot; Stephan Alaniz (University of Amsterdam)</i></p>

      <p><b>Kalman Filter, Sensor Fusion, and Constrained Regression: Equivalences and Insights</b><br><i>David Farrow (Carnegie Mellon University) &middot; Maria Jahja (Carnegie Mellon University) &middot; Roni Rosenfeld (Carnegie Mellon University) &middot; Ryan Tibshirani (Carnegie Mellon University)</i></p>

      <p><b>Near Neighbor: Who is the Fairest of Them All?</b><br><i>Sepideh Mahabadi (Toyota Technological Institute at Chicago) &middot; Sariel Har-Peled (University of Illinois at Urbana-Champaign)</i></p>

      <p><b>Outlier-robust estimation of a sparse linear model using $\ell_1$-penalized Huber&#39;s $M$-estimator</b><br><i>Arnak Dalalyan (ENSAE ParisTech) &middot; Philip Thompson (ENSAE ParisTech - Centre for Research in Economics and Statistic)</i></p>

      <p><b>Learning nonlinear level sets for dimensionality reduction in function approximation</b><br><i>Guannan Zhang (Oak Ridge National Laboratory) &middot; Jiaxin Zhang (Oak Ridge National Laboratory) &middot; Jacob Hinkle (Oak Ridge National Lab)</i></p>

      <p><b>Assessing Social and Intersectional Biases in Contextualized Word Representations</b><br><i>Yi Chern Tan (Yale University) &middot; L. Elisa Celis (Yale University)</i></p>

      <p><b>Online Convex Matrix Factorization with Representative Regions</b><br><i>Jianhao Peng (University of Illinois at Urbana Champaign) &middot; Olgica Milenkovic (University of Illinois at Urbana-Champaign) &middot; Abhishek Agarwal (University of Illinois at Urbana Champaign)</i></p>

      <p><b>Self-supervised GAN: Analysis and Improvement with Multi-class Minimax Game</b><br><i>Ngoc-Trung Tran (Singapore University of Technology and Design) &middot; Viet-Hung Tran (Singapore University of Technology and Design) &middot; Bao-Ngoc Nguyen (Singapore University of Technology and Design) &middot; Linxiao Yang (University of Electronic Science  and Technology of China; Singapore University of Technology and Design) &middot; Ngai-Man Cheung (Singapore University of Technology and Design)</i></p>

      <p><b>Simultaneous Matching and Ranking as end-to-end Deep Classification: A Case study of Information Retrieval with 50M Documents</b><br><i>Tharun Kumar Reddy Medini (Rice University) &middot; Qixuan Huang (Rice University) &middot; Yiqiu Wang (Massachusetts Institute of Technology) &middot; Vijai Mohan (www.amazon.com) &middot; Anshumali Shrivastava (Rice University)</i></p>

      <p><b>A Fourier Perspective on Model Robustness in Computer Vision</b><br><i>Dong Yin (UC Berkeley) &middot; Raphael Gontijo Lopes (Google Brain) &middot; Ekin Dogus Cubuk (Google Brain) &middot; Justin Gilmer (Google Brain) &middot; Jon Shlens (Google Research)</i></p>

      <p><b>The continuous Bernoulli: fixing a pervasive error in variational autoencoders</b><br><i>Gabriel Loaiza-Ganem (Columbia University) &middot; John Cunningham (University of Columbia)</i></p>

      <p><b>Privacy Amplification by Mixing and Diffusion Mechanisms</b><br><i>Borja Balle (Amazon Research Cambridge) &middot; Gilles Barthe (Max Planck Institute) &middot; Marco Gaboardi (Univeristy at Buffalo) &middot; Joseph Geumlek (UCSD)</i></p>

      <p><b>Variance Reduction in Bipartite Experiments through Correlation Clustering</b><br><i>Jean Pouget-Abadie (Harvard University) &middot; Kevin Aydin (Google) &middot; Warren Schudy (Google) &middot; Kay Brodersen (Google) &middot; Vahab Mirrokni (Google Research NYC)</i></p>

      <p><b>Gossip-based Actor-Learner Architectures for Deep Reinforcement Learning</b><br><i>Mahmoud Assran (McGill University / Facebook AI Research) &middot; Joshua Romoff (McGill University) &middot; Nicolas Ballas (Facebook FAIR) &middot; Joelle Pineau (Facebook) &middot; Mike Rabbat (Facebook FAIR)</i></p>

      <p><b>Metalearned Neural Memory</b><br><i>Tsendsuren Munkhdalai (Microsoft Research) &middot; Alessandro Sordoni (Microsoft Research Montreal) &middot; TONG WANG (Microsoft Research Montreal) &middot; Adam Trischler (Microsoft)</i></p>

      <p><b>Learning Multiple Markov Chains via Adaptive Allocation</b><br><i>Mohammad Sadegh Talebi (Inria) &middot; Odalric-Ambrym Maillard (INRIA)</i></p>

      <p><b>Diffusion Improves Graph Learning</b><br><i>Johannes Klicpera (Technical University of Munich) &middot; Stefan Weißenberger (Technical University of Munich) &middot; Stephan Günnemann (Technical University of Munich)</i></p>

      <p><b>Deep Random Splines for Point Process Intensity Estimation of Neural Population Data</b><br><i>Gabriel Loaiza-Ganem (Columbia University) &middot; John Cunningham (University of Columbia) &middot; Sean Perkins (Columbia University) &middot; Karen Schroeder (Columbia University) &middot; Mark Churchland (Columbia University)</i></p>

      <p><b>Variational Bayes under Model Misspecification</b><br><i>Yixin Wang (Columbia University) &middot; David Blei (Columbia University)</i></p>

      <p><b>On the Importance of Initialization in Optimization for Deep Linear Neural Networks</b><br><i>Lei Wu (Princeton University) &middot; Qingcan Wang (PACM, Princeton University) &middot; Chao Ma (Princeton University)</i></p>

      <p><b>On Differentially Private Graph Sparsification and Applications</b><br><i>Raman Arora (Johns Hopkins University) &middot; Jalaj Upadhyay (Johns Hopkins University)</i></p>

      <p><b>Manifold denoising by Nonlinear Robust Principal Component Analysis</b><br><i>Rongrong Wang (Michigan State University) &middot; Ming Yan (Michigan State University) &middot; He Lyu (Michigan State University) &middot; Yuying Xie (Michigan State University) &middot; Ningyu Sha (MSU) &middot; Shuyang Qin (Michigan State University)</i></p>

      <p><b>Near-Optimal Reinforcement Learning in Dynamic Treatment Regimes</b><br><i>Junzhe Zhang (Purdue University) &middot; Elias Bareinboim (Purdue)</i></p>

      <p><b>ODE2VAE: Deep generative second order ODEs with Bayesian neural networks</b><br><i>Cagatay Yildiz (Aalto University) &middot; Markus Heinonen (Aalto University) &middot; Harri Lahdesmaki (Aalto University)</i></p>

      <p><b>Optimal Sampling and Clustering in the Stochastic Block Model</b><br><i>Se-Young Yun (KAIST) &middot; Alexandre Proutiere (KTH)</i></p>

      <p><b>Recurrent Kernel Networks</b><br><i>Dexiong Chen (Inria) &middot; Laurent Jacob (CNRS) &middot; Julien Mairal (Inria)</i></p>

      <p><b>Cold Case: The Lost MNIST Digits</b><br><i>Chhavi Yadav (Walmart Labs, NYU) &middot; Leon Bottou (Facebook AI Research)</i></p>

      <p><b>Hierarchical Optimal Transport for Multimodal Distribution Alignment</b><br><i>John Lee (Georgia Institute of Technology) &middot; Max Dabagia (Georgia Institute of Technology) &middot; Eva Dyer (Georgia Tech) &middot; Christopher Rozell (Georgia Institute of Technology)</i></p>

      <p><b>Exploration via Hindsight Goal Generation</b><br><i>Zhizhou Ren (Tsinghua University) &middot; Kefan Dong (Tsinghua University) &middot; Yuan Zhou (Indiana University Bloomington) &middot; Qiang Liu (UT Austin) &middot; Jian Peng (University of Illinois at Urbana-Champaign)</i></p>

      <p><b>Shaping Belief States with Generative Environment Models for RL</b><br><i>Karol Gregor (DeepMind) &middot; Danilo Jimenez Rezende (Google DeepMind) &middot; Frederic Besse (DeepMind) &middot; Yan Wu (DeepMind) &middot; Hamza Merzic (Deepmind) &middot; Aaron van den Oord (Google Deepmind)</i></p>

      <p><b>Globally Optimal Learning for Structured Elliptical Losses</b><br><i>Yoav Wald (Hebrew University) &middot; Nofar Noy (Hebrew University) &middot; Gal Elidan (Google) &middot; Ami Wiesel (Google Research and The Hebrew University of Jerusalem, Israel)</i></p>

      <p><b>Object landmark discovery through unsupervised adaptation</b><br><i>Enrique Sanchez (Samsung AI Centre) &middot; Georgios Tzimiropoulos (University of Nottingham)</i></p>

      <p><b>Specific and Shared Causal Relation Modeling and Mechanism-based Clustering</b><br><i>Biwei Huang (Carnegie Mellon University) &middot; Kun Zhang (CMU) &middot; Pengtao Xie (Petuum / CMU) &middot; Mingming Gong (University of Melbourne) &middot; Eric Xing (Petuum Inc.) &middot; Clark Glymour (Carnegie Mellon University)</i></p>

      <p><b>Search-Guided, Lightly-Supervised Training of Structured Prediction Energy Networks</b><br><i>Amirmohammad Rooshenas (University of Massachusetts, Amherst) &middot; Dongxu Zhang (University of Massachusetts Amherst) &middot; Gopal Sharma (University of Massachusetts Amherst) &middot; Andrew McCallum (UMass Amherst)</i></p>

      <p><b>Accelerating Rescaled Gradient Descent: Fast Optimization of Smooth Functions</b><br><i>Ashia Wilson (UC Berkeley) &middot; Lester Mackey (Microsoft Research) &middot; Andre Wibisono ()</i></p>

      <p><b>RUDDER: Return Decomposition for Delayed Rewards</b><br><i>José Arjona-Medina (LIT AI Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria) &middot; Michael Gillhofer (LIT AI Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria) &middot; Michael Widrich (LIT AI Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria) &middot; Thomas Unterthiner (LIT AI Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria) &middot; Johannes Brandstetter (LIT AI Lab / University Linz) &middot; Sepp Hochreiter (LIT AI Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria)</i></p>

      <p><b>Graph Normalizing Flows</b><br><i>Jenny Liu (University of Toronto) &middot; Aviral Kumar (UC Berkeley) &middot; Jimmy Ba (University of Toronto / Vector Institute) &middot; Jamie Kiros (Google Inc.) &middot; Kevin Swersky (Google)</i></p>

      <p><b>Explanations can be manipulated and geometry is to blame</b><br><i>Ann-Kathrin Dombrowski (TU Berlin) &middot; Maximillian Alber (TU Berlin) &middot; Christopher Anders (Technische Universität Berlin) &middot; Marcel Ackermann (HHI) &middot; Klaus-Robert Müller (TU Berlin) &middot; Pan Kessel (TU Berlin)</i></p>

      <p><b>Communication trade-offs for synchronized distributed SGD with large step size</b><br><i>Aymeric Dieuleveut (EPFL) &middot; Kshitij Patel (Indian Institute of Technology Kanpur)</i></p>

      <p><b>Non-normal Recurrent Neural Network (nnRNN): learning long time dependencies while improving expressivity with transient dynamics</b><br><i>Giancarlo Kerg (MILA) &middot; Kyle Goyette (University of Montreal) &middot; Maximilian Puelma Touzel (Mila) &middot; Gauthier Gidel (Mila) &middot; Eugene Vorontsov (Polytechnique Montreal) &middot; Yoshua Bengio (Mila) &middot; Guillaume Lajoie (Université de Montréal / Mila)</i></p>

      <p><b>No-Regret Learning in Unknown Games with Correlated Payoffs</b><br><i>Pier Giuseppe Sessa (ETH Zürich) &middot; Ilija Bogunovic (ETH Zurich) &middot; Maryam Kamgarpour (ETH Zürich) &middot; Andreas Krause (ETH Zurich)</i></p>

      <p><b>Alleviating Label Switching with Optimal Transport</b><br><i>Pierre Monteiller (ENS Ulm ) &middot; Sebastian Claici (MIT) &middot; Edward Chien (Massachusetts Institute of Technology) &middot; Farzaneh Mirzazadeh (IBM Research, MIT-IBM Watson AI Lab) &middot; Justin M Solomon (MIT) &middot; Mikhail Yurochkin (IBM Research, MIT-IBM Watson AI Lab)</i></p>

      <p><b>Paraphrase Generation with Latent Bag of Words</b><br><i>Yao Fu (Columbia University) &middot; Yansong Feng (Peking University) &middot; John Cunningham (University of Columbia)</i></p>

      <p><b>An Algorithmic Framework For Differentially Private Data Analysis on Trusted Processors</b><br><i>Janardhan Kulkarni (MSR, Redmond) &middot; Olga Ohrimenko (Microsoft Research) &middot; Bolin Ding (Alibaba Group) &middot; Sergey Yekhanin (Microsoft) &middot; Joshua  Allen (Microsoft) &middot; Harsha Nori (Microsoft)</i></p>

      <p><b>Compacting, Picking and Growing for Unforgetting Continual Learning</b><br><i>Ching-Yi Hung (Academia Sinica) &middot; Cheng-Hao Tu (Academia Sinica) &middot; Cheng-En Wu (Academia Sinica) &middot; Chien-Hung Chen (Academia Sinica) &middot; Yi-Ming Chan (Academia Sinica) &middot; Chu-Song Chen (Academia Sinica)</i></p>

      <p><b>Approximating Interactive Human Evaluation withSelf-Play for Open-Domain Dialog Systems</b><br><i>Asma Ghandeharioun (MIT) &middot; Judy Hanwen Shen (Massachusetts Institute of Technology) &middot; Natasha Jaques (MIT) &middot; Craig Ferguson (MIT) &middot; Noah Jones (MIT) &middot; Agata Garcia (Massachusetts Institute of Technology) &middot; Rosalind Picard (MIT Media Lab)</i></p>

      <p><b> A New Distribution on the Simplex with Auto-Encoding Applications</b><br><i>Andrew Stirn (Columbia University) &middot; Tony Jebara (Netflix) &middot; David Knowles (Columbia University)</i></p>

      <p><b>AutoPrun: Automatic Network Pruning by Regularizing Auxiliary Parameters</b><br><i>XIA XIAO (University of Connecticut) &middot; Zigeng Wang (University of Connecticut) &middot; Sanguthevar Rajasekaran (University of Connecticut)</i></p>

      <p><b>A neurally plausible model learns successor representations in partially observable environments</b><br><i>Eszter Vértes (Gatsby Unit, UCL) &middot; Maneesh Sahani (Gatsby Unit, UCL)</i></p>

      <p><b>Learning about an exponential amount of conditional distributions</b><br><i>Mohamed Belghazi (University of Montreal) &middot; Maxime Oquab (Facebook AI Research) &middot; David Lopez-Paz (Facebook AI Research)</i></p>

      <p><b>Towards modular and programmable architecture search</b><br><i>Renato Negrinho (Carnegie Mellon University) &middot; Matthew Gormley (Carnegie Mellon University) &middot; Geoffrey Gordon (MSR Montréal & CMU) &middot; Darshan Patil (Carnegie Mellon University) &middot; Nghia Le (Carnegie Mellon University) &middot; Daniel Ferreira (TU Wien)</i></p>

      <p><b>Towards Hardware-Aware Tractable Learning of Probabilistic Models</b><br><i>Laura I. Galindez Olascoaga (KU Leuven) &middot; Wannes Meert (K.U.Leuven) &middot; Marian Verhelst (KU Leuven) &middot; Guy Van den Broeck (UCLA)</i></p>

      <p><b>On Robustness to Adversarial Examples and Polynomial Optimization</b><br><i>Pranjal Awasthi (Rutgers University/Google) &middot; Abhratanu Dutta (Northwestern University) &middot; Aravindan Vijayaraghavan (Northwestern University)</i></p>

      <p><b>Rand-NSG: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node</b><br><i>Suhas Jayaram Subramanya (Microsoft Research India) &middot; Devvrit Lnu (BITS Pilani) &middot; Harsha Vardhan Simhadri (Microsoft Research India) &middot; Ravishankar Krishnawamy (Microsoft Research India)</i></p>

      <p><b>A Solvable High-Dimensional Model of GAN</b><br><i>Chuang Wang (Institute of Automation, Chinese Academy of Sciences)</i></p>

      <p><b>Using Embeddings to Correct for Unobserved Confounding in Networks</b><br><i>Victor Veitch (Columbia University) &middot; Yixin Wang (Columbia University) &middot; David Blei (Columbia University)</i></p>

      <p><b>PolyTree framework for tree ensemble analysis</b><br><i>Igor E. Kuralenok (Experts League Ltd.) &middot; Vasilii Ershov (Yandex) &middot; Igor Labutin (Saint Petersburg campus of National Research University Higher School of Economics)</i></p>

      <p><b>Bayesian Optimization under Heavy-tailed Payoffs</b><br><i>Sayak Ray Chowdhury (Indian Institute of Science) &middot; Aditya Gopalan (Indian Institute of Science)</i></p>

      <p><b>Combining Generative and Discriminative Models for Hybrid Inference</b><br><i>Victor Garcia Satorras (UPC) &middot; Max Welling (University of Amsterdam / Qualcomm AI Research) &middot; Zeynep Akata (University of Amsterdam)</i></p>

      <p><b>A Graph Theoretic Additive Approximation of Optimal Transport</b><br><i>Nathaniel Lahn (Virginia Tech) &middot; Deepika Mulchandani (Virginia Tech) &middot; Sharath Raghvendra (Virginia Tech)</i></p>

      <p><b>Adversarial Robustness through Local Linearization</b><br><i>Chongli Qin (DeepMind) &middot; James Martens (DeepMind) &middot; Sven Gowal (DeepMind) &middot; Dilip Krishnan (Google) &middot; Krishnamurthy Dvijotham (DeepMind) &middot; Alhussein Fawzi (DeepMind) &middot; Soham De (DeepMind) &middot; Robert Stanforth (DeepMind) &middot; Pushmeet Kohli (DeepMind)</i></p>

      <p><b>Sampled softmax with random Fourier features</b><br><i>Ankit Singh Rawat (Google Research) &middot; Jiecao Chen (Indiana University Bloomington) &middot; Felix Xinnan Yu (Google Research) &middot; Ananda Theertha Suresh (Google) &middot; Sanjiv Kumar (Google Research)</i></p>

      <p><b>Semi-flat minima and saddle points by embedding neural networks to overparameterization</b><br><i>Kenji Fukumizu (Institute of Statistical Mathematics / Preferred Networks / RIKEN AIP) &middot; Shoichiro Yamaguchi (Preferred Networks) &middot; Yoh-ichi Mototake (Institute of Statistical Mathematics) &middot; Mirai Tanaka (The Institute of Statistical Mathematics / RIKEN)</i></p>

      <p><b>Learning Fairness in Multi-Agent Systems</b><br><i>Jiechuan Jiang (Peking University) &middot; Zongqing Lu (Peking University)</i></p>

      <p><b>Primal-Dual Block Frank-Wolfe</b><br><i>Qi Lei (University of Texas at Austin) &middot; JIACHENG ZHUO (University of Texas at Austin) &middot; Constantine Caramanis (UT Austin) &middot; Inderjit S Dhillon (UT Austin & Amazon) &middot; Alexandros Dimakis (University of Texas, Austin)</i></p>

      <p><b>GOT: An Optimal Transport framework for Graph comparison</b><br><i>Hermina Petric Maretic (Ecole Polytechnique Fédérale de Lausanne) &middot; Mireille El Gheche (EPFL) &middot; Giovanni Chierchia (ESIEE Paris) &middot; Pascal Frossard (EPFL)</i></p>

      <p><b>On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks</b><br><i>Sunil Thulasidasan (Los Alamos National Laboratory) &middot; Gopinath Chennupati (Los Alamos National Laboratory) &middot; Jeff Bilmes (University of Washington, Seattle) &middot; Tanmoy Bhattacharya (Los Alamos National Laboratory) &middot; Sarah Michalak (Los Alamos National Laboratory)</i></p>

      <p><b>Complexity of Highly Parallel Non-Smooth Convex Optimization</b><br><i>Sebastien Bubeck (Microsoft Research) &middot; Qijia Jiang (Stanford University) &middot; Yin-Tat Lee () &middot; Yuanzhi Li (Princeton) &middot; Aaron Sidford (Stanford)</i></p>

      <p><b>Inverting Deep Generative models, One layer at a time</b><br><i>Qi Lei (University of Texas at Austin) &middot; Ajil Jalal (University of Texas at Austin) &middot; Inderjit S Dhillon (UT Austin & Amazon) &middot; Alexandros Dimakis (University of Texas, Austin)</i></p>

      <p><b>Calculating Optimistic Likelihoods Using (Geodesically) Convex Optimization</b><br><i>Viet Anh Nguyen (EPFL) &middot; Soroosh Shafieezadeh Abadeh (EPFL) &middot; Man-Chung Yue (The Hong Kong Polytechnic University) &middot; Daniel Kuhn (EPFL) &middot; Wolfram Wiesemann (Imperial College)</i></p>

      <p><b>The Implicit Metropolis-Hastings Algorithm</b><br><i>Kirill Neklyudov (Samsung AI Center, Moscow) &middot; Evgenii Egorov (Skolkovo Institute of Science and Technology) &middot; Dmitry Vetrov (Higher School of Economics, Samsung AI Center, Moscow)</i></p>

      <p><b>An  Inexact Augmented Lagrangian Framework for Nonconvex Optimization with Nonlinear Constraints</b><br><i>Mehmet Fatih SAHIN (École polytechnique fédérale de Lausanne) &middot; Armin eftekhari (EPFL) &middot; Ahmet Alacaoglu (EPFL) &middot; Fabian Latorre Gomez (EPFL) &middot; Volkan Cevher (EPFL)</i></p>

      <p><b>Generalization in Reinforcement Learning with Selective Noise Injection and Information Bottleneck</b><br><i>Maximilian Igl (University of Oxford) &middot; Kamil Ciosek (Microsoft) &middot; Yingzhen Li (Microsoft Research Cambridge) &middot; Sebastian Tschiatschek (Microsoft Research) &middot; Cheng Zhang (Microsoft) &middot; Sam Devlin (Microsoft Research) &middot; Katja Hofmann (Microsoft Research)</i></p>

      <p><b>Can you trust your model&#39;s uncertainty?  Evaluating predictive uncertainty under dataset shift</b><br><i>Jasper Snoek (Google Brain) &middot; Yaniv Ovadia (Google Inc) &middot; Emily Fertig (Google Brain) &middot; Balaji Lakshminarayanan (Google DeepMind) &middot; Sebastian Nowozin (Google Research) &middot; D. Sculley (Google Research) &middot; Joshua Dillon (Google) &middot; Jie Ren (Google Inc.) &middot; Zachary Nado (Google Inc.)</i></p>

      <p><b>Accurate Layerwise Interpretable Competence Estimation</b><br><i>Vickram Rajendran (JHU Applied Physics Laboratory) &middot; Will LeVine (Rice University)</i></p>

      <p><b>A New Perspective on Pool-Based Active Classification and False-Discovery Control</b><br><i>Lalit Jain (University of Washington) &middot; Kevin Jamieson (U Washington)</i></p>

      <p><b>A First-Order Approach to Accelerated Value Iteration</b><br><i>Julien Grand Clement (IEOR Department, Columbia University) &middot; Vineet Goyal (Columbia University)</i></p>

      <p><b>Defending Neural Backdoors via Generative Distribution Modeling</b><br><i>Ximing Qiao (Duke University) &middot; Yukun Yang (Duke University) &middot; Hai Li (Duke University)</i></p>

      <p><b>Are Sixteen Heads Really Better than One?</b><br><i>Paul Michel (Carnegie Mellon University, Language Technologies Institute) &middot; Omer Levy (Facebook) &middot; Graham Neubig (Carnegie Mellon University)</i></p>

      <p><b>Multi-resolution Multi-task Gaussian Processes</b><br><i>Oliver Hamelijnck (The Alan Turing Institute) &middot; Theodoros Damoulas (University of Warwick        The Alan Turing Institute) &middot; Kangrui Wang (The Alan Turing Institute) &middot; Mark Girolami (Imperial College London)</i></p>

      <p><b>Variational Bayesian Optimal Experimental Design</b><br><i>Adam Foster (University of Oxford) &middot; Martin Jankowiak (Uber AI Labs) &middot; Eli Bingham (Uber AI Labs) &middot; Paul Horsfall (Uber AI Labs) &middot; Yee Whye Teh (University of Oxford, DeepMind) &middot; Tom Rainforth (University of Oxford) &middot; Noah Goodman (Stanford University)</i></p>

      <p><b>Universal Approximation of Input-Output Maps by Temporal Convolutional Nets</b><br><i>Joshua Hanson (University of Illinois) &middot; Maxim Raginsky (University of Illinois at Urbana-Champaign)</i></p>

      <p><b>Provable Certificates for Adversarial Examples: Fitting a Ball in the Union of Polytopes</b><br><i>Matt Jordan (UT Austin) &middot; justin lewis (University of Texas at Austin) &middot; Alexandros Dimakis (University of Texas, Austin)</i></p>

      <p><b>Reinforcement Learning with Convex Constraints </b><br><i>Seyed Sobhan Mir Yoosefi (Princeton University) &middot; Kianté Brantley (The University of Maryland College Park) &middot; Hal Daume III (Microsoft Research &      University of Maryland) &middot; Miro Dudik (Microsoft Research) &middot; Robert Schapire (MIcrosoft Research)</i></p>

      <p><b>User-Specified Local Differential Privacy in Unconstrained Adaptive Online Learning</b><br><i>Dirk van der Hoeven (Leiden University)</i></p>

      <p><b>Stochastic Bandits with Context Distributions</b><br><i>Johannes Kirschner (ETH Zurich) &middot; Andreas Krause (ETH Zurich)</i></p>

      <p><b>Inducing brain-relevant bias in natural language processing models</b><br><i>Dan Schwartz (Carnegie Mellon University) &middot; Mariya Toneva (Carnegie Mellon University) &middot; Leila Wehbe (Carnegie Mellon University)</i></p>

      <p><b>Using a Logarithmic Mapping to Enable Lower Discount Factors in Reinforcement Learning</b><br><i>Harm Van Seijen (Microsoft Research) &middot; Mehdi Fatemi (Microsoft Research) &middot; Arash Tavakoli (Imperial College London)</i></p>

      <p><b>Recovering Bandits</b><br><i>Ciara Pike-Burke (Universitat Pompeu Fabra) &middot; Steffen Grunewalder (Lancaster)</i></p>

      <p><b>Computing Linear Restrictions of Neural Networks</b><br><i>Matthew Sotoudeh (University of California, Davis) &middot; Aditya Thakur (University of California, Davis)</i></p>

      <p><b>Learning Positive Functions with Pseudo Mirror Descent</b><br><i>Yingxiang Yang (University of Illinois at Urbana Champaign) &middot; Haoxiang Wang (University of Illinois, Urbana-Champaign) &middot; Negar Kiyavash (Georgia Institute of Technology) &middot; Niao He (UIUC)</i></p>

      <p><b>Correlation Priors for Reinforcement Learning</b><br><i>Bastian Alt (Technische Universität Darmstadt) &middot; Adrian Šošić (Technische Universität Darmstadt) &middot; Heinz Koeppl (Technische Universität Darmstadt)</i></p>

      <p><b>Fast, Provably convergent IRLS Algorithm for p-norm Linear Regression</b><br><i>Deeksha Adil (University of Toronto) &middot; Richard Peng (Georgia Tech / MSR Redmond) &middot; Sushant Sachdeva (Yale University)</i></p>

      <p><b>A Similarity-preserving Network Trained on Transformed Images Recapitulates Salient Features of the Fly Motion Detection Circuit</b><br><i>Yanis Bahroun (Flatiron institute) &middot; Dmitri Chklovskii (Flatiron Institute, Simons Foundation) &middot; Anirvan Sengupta (Rutgers University)</i></p>

      <p><b>Differentially Private Covariance Estimation</b><br><i>Kareem Amin (Google Research) &middot; Travis Dick (Carnegie Mellon University) &middot; Alex Kulesza (Google) &middot; Andres Munoz (Google) &middot; Sergei Vassilvitskii (Google)</i></p>

      <p><b>Outlier Detection and Robust PCA Using a Convex Measure of Innovation</b><br><i>Mostafa Rahmani (Baidu Research) &middot; Ping Li (Baidu Research USA)</i></p>

      <p><b>Integrating mechanistic and structural causal models enables counterfactual inference in complex systems</b><br><i>Robert Ness (Gamalon) &middot; Kaushal Paneri (Northeastern University) &middot; Olga Vitek (Northeastern University)</i></p>

      <p><b>Are Disentangled Representations Helpful for Abstract Visual Reasoning?</b><br><i>Sjoerd van Steenkiste (The Swiss AI Lab - IDSIA) &middot; Francesco Locatello (ETH Zürich - MPI Tübingen) &middot; Jürgen Schmidhuber (Swiss AI Lab, IDSIA (USI & SUPSI) - NNAISENSE) &middot; Olivier Bachem (Google Brain)</i></p>

      <p><b>PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization</b><br><i>Thijs Vogels (EPFL) &middot; Sai Praneeth Reddy Karimireddy (EPFL) &middot; Martin Jaggi (EPFL)</i></p>

      <p><b>Stochastic Frank-Wolfe for Composite Convex Minimization</b><br><i>Francesco Locatello (ETH Zürich - MPI Tübingen) &middot; Alp Yurtsever (EPFL) &middot; Olivier Fercoq (Telecom ParisTech) &middot; Volkan Cevher (EPFL)</i></p>

      <p><b>Consistent Constraint-Based Causal Structure Learning </b><br><i>Honghao Li (Institut Curie) &middot; Vincent Cabeli (Institut Curie) &middot; Nadir Sella (Institut Curie) &middot; Herve Isambert (Institut Curie)</i></p>

      <p><b>Unsupervised Discovery of Temporal Structure in Noisy Data with Dynamical Components Analysis</b><br><i>David Clark (Lawrence Berkeley National Laboratory) &middot; Jesse Livezey (Lawrence Berkeley National Laboratory) &middot; Kristofer Bouchard (Lawrence Berkeley National Laboratory)</i></p>

      <p><b>Sample Efficient Active Learning of Causal Trees</b><br><i>Kristjan Greenewald (IBM Research) &middot; Dmitriy Katz (IBM Research) &middot; Karthikeyan Shanmugam (IBM Research, NY) &middot; Sara Magliacane (IBM Research AI) &middot; Murat Kocaoglu (MIT-IBM Watson AI Lab) &middot; Enric Boix Adsera (MIT) &middot; Guy Bresler (MIT)</i></p>

      <p><b>Efficient Neural Architecture Transformation Search in Channel-Level for Object Detection</b><br><i>Junran Peng (CASIA) &middot; Ming Sun (sensetime.com) &middot; ZHAO-XIANG ZHANG (Chinese Academy of Sciences, China) &middot; Tieniu Tan (Chinese Academy of Sciences) &middot; Junjie Yan (Sensetime Group Limited)</i></p>

      <p><b>Robust Attribution Regularization</b><br><i>Jiefeng Chen (University of Wisconsin-Madison) &middot; Xi Wu (Google) &middot; Vaibhav Rastogi (University of Wisconsin-Madison) &middot; Yingyu Liang (University of Wisconsin Madison) &middot; Somesh Jha (University of Wisconsin, Madison)</i></p>

      <p><b>Computational Mirrors: Blind Inverse Light Transport by Deep Matrix Factorization</b><br><i>Miika Aittala (MIT) &middot; Prafull Sharma (MIT) &middot; Lukas Murmann (Massachusetts Institute of Technology) &middot; Adam Yedidia (Massachusetts Institute of Technology) &middot; Gregory Wornell (MIT) &middot; Bill Freeman (MIT/Google) &middot; Fredo Durand (MIT)</i></p>

      <p><b>When to use parametric models in reinforcement learning?</b><br><i>Hado van Hasselt (DeepMind) &middot; Matteo Hessel (Google DeepMind) &middot; John Aslanides (DeepMind)</i></p>

      <p><b>General E(2)-Equivariant Steerable CNNs</b><br><i>Gabriele Cesa (University of Amsterdam) &middot; Maurice Weiler (University of Amsterdam)</i></p>

      <p><b>Characterization and Learning of Causal Graphs with Latent Variables from Soft Interventions</b><br><i>Murat Kocaoglu (MIT-IBM Watson AI Lab) &middot; Karthikeyan Shanmugam (IBM Research, NY) &middot; Amin Jaber (Purdue University) &middot; Elias Bareinboim (Purdue)</i></p>

      <p><b>Structure Learning with Side Information: Sample Complexity</b><br><i>Saurabh Sihag (Rensselaer Polytechnic Institute) &middot; Ali Tajer (Rensselaer Polytechnic Institute)</i></p>

      <p><b>Untangling in Invariant Speech Recognition </b><br><i>Cory Stephenson (Intel) &middot; Jenelle Feather (MIT) &middot; Suchismita Padhy (Intel AI Lab) &middot; Oguz Elibol (Intel Nervana) &middot; Hanlin Tang (Intel AI Products Group) &middot; Josh McDermott (Massachusetts Institute of Technology) &middot; Sueyeon Chung (MIT)</i></p>

      <p><b>Flexible information routing in neural populations through stochastic comodulation</b><br><i>Caroline Haimerl (New York University) &middot; Cristina Savin (NYU) &middot; Eero Simoncelli (HHMI / New York University)</i></p>

      <p><b>Generalization Bounds in the Predict-then-Optimize Framework</b><br><i>Othman El Balghiti (Columbia University) &middot; Adam Elmachtoub (Columbia University) &middot; Paul Grigas (UC Berkeley) &middot; Ambuj Tewari (University of Michigan)</i></p>

      <p><b>Categorized Bandits</b><br><i>Matthieu Jedor (ENS Paris-Saclay & Cdiscount) &middot; Vianney Perchet (ENS Paris-Saclay & Criteo AI Lab) &middot; Jonathan Louedec (Cdiscount)</i></p>

      <p><b>Worst-Case Regret Bounds for Exploration via Randomized Value Functions</b><br><i>Daniel Russo (Columbia University)</i></p>

      <p><b>Efficient characterization of electrically evoked responses for neural interfaces</b><br><i>Nishal Shah (Stanford University) &middot; Sasidhar Madugula (Stanford University) &middot; Pawel Hottowy (AGH University of Science and Technology in Kraków) &middot; Alexander Sher (Santa Cruz Institute for Particle Physics, University of California, Santa Cruz) &middot; Alan Litke (Santa Cruz Institute for Particle Physics, University of California, Santa Cruz) &middot; Liam Paninski (Columbia University) &middot; E.J. Chichilnisky (Stanford University)</i></p>

      <p><b>Differentially Private Distributed Data Summarization under Covariate Shift</b><br><i>Kanthi K Sarpatwar (IBM T. J. Watson Research Center) &middot; Karthikeyan Shanmugam (IBM Research, NY) &middot; Venkata Sitaramagiridharganesh Ganapavarapu (IBM Research) &middot; Ashish Jagmohan (IBM Research) &middot; Roman Vaculin (IBM Research)</i></p>

      <p><b>Hamiltonian descent for composite objectives</b><br><i>Brendan O'Donoghue (Google DeepMind) &middot; Chris J. Maddison (Institute for Advanced Study, Princeton)</i></p>

      <p><b>Implicit Regularization of Accelerated Methods in Hilbert Spaces</b><br><i>Nicolò Pagliana (Università degli studi di Genova (DIMA)) &middot; Lorenzo Rosasco (University of Genova- MIT - IIT)</i></p>

      <p><b>Non-Asymptotic Pure Exploration by Solving Games</b><br><i>Rémy Degenne (Centrum Wiskunde & Informatica, Amsterdam) &middot; Wouter Koolen (Centrum Wiskunde & Informatica, Amsterdam) &middot; Pierre Ménard (Institut de Mathématiques de Toulouse)</i></p>

      <p><b>Implicit Posterior Variational Inference for Deep Gaussian Processes</b><br><i>Haibin YU (National University of Singapore) &middot; Yizhou Chen (National University of Singapore) &middot; Bryan Kian Hsiang Low (National University of Singapore) &middot; Patrick Jaillet (MIT)</i></p>

      <p><b>Deep Multi-State Dynamic Recurrent Neural Networks Operating on Wavelet Based Neural Features for Robust Brain Machine Interfaces</b><br><i>Benyamin Allahgholizadeh Haghi (California Institute of Technology) &middot; Spencer Kellis (California Institute of Technology) &middot; Sahil Shah (California Institute of Technology) &middot; Maitreyi Ashok (California Institute of Technology) &middot; Luke Bashford (California Institute of Technology) &middot; Daniel Kramer (University of Southern California) &middot; Brian Lee (University of Southern California) &middot; Charles Liu (University of Southern California) &middot; Richard Andersen (California Institute of Technology) &middot; Azita Emami (California Institute of Technology)</i></p>

      <p><b>Censored Semi-Bandits: A Framework for Resource Allocation with Censored Feedback</b><br><i>Arun Verma (IIT Bombay) &middot; Manjesh K Hanawal (Indian Institute of Technology Bombay) &middot; Arun Rajkumar (Xerox Research Center, India.) &middot; Raman Sankaran (LinkedIn)</i></p>

      <p><b>Cormorant: Covariant Molecular Neural Networks</b><br><i>Brandon Anderson (University of Chicago) &middot; Truong Son Hy (The University of Chicago) &middot; Risi Kondor (U. Chicago)</i></p>

      <p><b>Reverse KL-Divergence Training of Prior Networks: Improved Uncertainty and Adversarial Robustness</b><br><i>Andrey Malinin (University of Cambridge) &middot; Mark Gales (University of Cambridge)</i></p>

      <p><b>Reflection Separation using a Pair of Unpolarized and Polarized Images</b><br><i>Youwei Lyu (Beijing University of Posts and Telecommunications) &middot; Zhaopeng Cui (ETH Zurich) &middot; Si Li (Beijing University of Posts and Telecommunications) &middot; Marc Pollefeys (ETH Zurich) &middot; Boxin Shi (Peking University)</i></p>

      <p><b>Policy Poisoning in Batch Reinforcement Learning and Control</b><br><i>Yuzhe Ma (University of Wisconsin-Madison) &middot; Xuezhou Zhang (UW-Madison) &middot; Wen Sun (Microsoft Research) &middot; Jerry Zhu (University of Wisconsin-Madison)</i></p>

      <p><b>Low-Complexity Nonparametric Bayesian Online Prediction with Universal Guarantees</b><br><i>Alix LHERITIER (Amadeus SAS) &middot; Frederic Cazals (Inria)</i></p>

      <p><b>Pure Exploration with Multiple Correct Answers</b><br><i>Rémy Degenne (Centrum Wiskunde & Informatica, Amsterdam) &middot; Wouter Koolen (Centrum Wiskunde & Informatica, Amsterdam)</i></p>

      <p><b>Explaining Landscape Connectivity of Low-cost Solutions for Multilayer Nets</b><br><i>Rohith Kuditipudi (Duke University) &middot; Xiang Wang (Duke University) &middot; HOLDEN LEE (Princeton) &middot; Yi Zhang (Princeton) &middot; Zhiyuan Li (Princeton University) &middot; Wei Hu (Princeton University) &middot; Rong Ge (Duke University) &middot; Sanjeev Arora (Princeton University)</i></p>

      <p><b>On the Benefits of Disentangled Representations</b><br><i>Francesco Locatello (ETH Zürich - MPI Tübingen) &middot; Gabriele Abbati (University of Oxford) &middot; Tom Rainforth (University of Oxford) &middot; Stefan Bauer (MPI for Intelligent Systems) &middot; Bernhard Schölkopf (MPI for Intelligent Systems) &middot; Olivier Bachem (Google Brain)</i></p>

      <p><b>Compiler Auto-Vectorization using Imitation Learning</b><br><i>Charith Mendis (MIT) &middot; Cambridge Yang (MIT) &middot; Yewen Pu (MIT) &middot; Dr.Saman Amarasinghe (Massachusetts institute of technology) &middot; Michael Carbin (MIT)</i></p>

      <p><b>A Generalized Algorithm for Multi-Objective RL and Policy Adaptation</b><br><i>Runzhe Yang (Princeton University) &middot; Xingyuan Sun (Princeton University) &middot; Karthik Narasimhan (Princeton University)</i></p>

      <p><b>Exact Gaussian Processes on a Million Data Points</b><br><i>Ke Wang (Cornell University) &middot; Geoff Pleiss (Cornell University) &middot; Jacob Gardner (Uber AI Labs) &middot; Stephen Tyree (NVIDIA) &middot; Kilian Weinberger (Cornell University) &middot; Andrew Wilson (Cornell University)</i></p>

      <p><b>Bayesian Layers: A Module for Neural Network Uncertainty</b><br><i>Dustin Tran (Google Brain) &middot; Mike Dusenberry (Google Brain) &middot; Mark van der Wilk (PROWLER.io) &middot; Danijar Hafner (Google)</i></p>

      <p><b>Learning Compositional Neural Programs with Recursive Tree Search and Planning</b><br><i>Thomas PIERROT (InstaDeep) &middot; Guillaume Ligner (InstaDeep) &middot; Scott Reed (Google DeepMind) &middot; Olivier Sigaud (Sorbonne University) &middot; Perrin Nicolas (ISIR) &middot; David Kas (InstaDeep) &middot; David Kas (InstaDeep) &middot; Karim Beguir (InstaDeep) &middot; Nando de Freitas (DeepMind)</i></p>

      <p><b>Nonparametric Contextual Bandits in Metric Spaces with Unknown Metric</b><br><i>Nirandika Wanigasekara (National University of Singapore) &middot; Christina Lee Yu (Cornell University)</i></p>

      <p><b>Qsparse-local-SGD: Distributed SGD with Quantization, Sparsification and Local Computations</b><br><i>Debraj Basu (University of California Los Angeles) &middot; Deepesh Data (UCLA) &middot; Can Karakus (Amazon Web Services) &middot; Suhas Diggavi (UCLA)</i></p>

      <p><b>Likelihood Ratios for Out-of-Distribution Detection</b><br><i>Jie Ren (Google Brain) &middot; Peter Liu (Google Brain) &middot; Emily Fertig (Google Brain) &middot; Jasper Snoek (Google Brain) &middot; Ryan  Poplin (Google) &middot; Mark Depristo (Google) &middot; Joshua Dillon (Google) &middot; Balaji Lakshminarayanan (Google DeepMind)</i></p>

      <p><b>Discrete Flows: Invertible Generative Models of Discrete Data</b><br><i>Dustin Tran (Google Brain) &middot; Keyon Vafa (Columbia University) &middot; Kumar Agrawal (Google AI Resident) &middot; Laurent Dinh (Google Research) &middot; Ben Poole (Google Brain)</i></p>

      <p><b>Mindreader: A Self Validation Network for Object-Level Human Attention Reasoning</b><br><i>Zehua Zhang (Indiana University Bloomington) &middot; Chen Yu (Indiana University) &middot; David Crandall (Indiana University)</i></p>

      <p><b>Model Selection for Contextual Bandits</b><br><i>Dylan Foster (MIT) &middot; Akshay Krishnamurthy (Microsoft) &middot; Haipeng Luo (University of Southern California)</i></p>

      <p><b>Sliced Gromov-Wasserstein</b><br><i>Vayer Titouan (IRISA) &middot; Rémi Flamary (Université Côte d'Azur, 3IA Côte d'Azur) &middot; Nicolas Courty (IRISA, Universite Bretagne-Sud) &middot; Romain Tavenard (LETG-Rennes /  IRISA-Obelix) &middot; Laetitia Chapel (IRISA)</i></p>

      <p><b>Towards Practical Alternating Least-Squares for CCA</b><br><i>Zhiqiang Xu (Baidu Inc.) &middot; Ping Li (Baidu Research USA)</i></p>

      <p><b>Deep Leakage from Gradients</b><br><i>Ligeng Zhu (Simon Fraser University) &middot; Zhijian Liu (MIT) &middot; Song Han (MIT)</i></p>

      <p><b>Invariance-inducing regularization using worst-case transformations suffices to boost accuracy and spatial robustness</b><br><i>Fanny Yang (Stanford) &middot; Zuowen Wang (ETH Zurich) &middot; Christina Heinze-Deml (ETH Zurich)</i></p>

      <p><b>Algorithm-Dependent Generalization Bounds for Overparameterized Deep Residual Networks</b><br><i>Spencer Frei (UCLA) &middot; Yuan Cao (UCLA) &middot; Quanquan Gu (UCLA)</i></p>

      <p><b>Value Function in Frequency Domain and Characteristic Value Iteration</b><br><i>Amir-massoud Farahmand (Vector Institute)</i></p>

      <p><b>Icebreaker: Efficient Information Acquisition with Active Learning</b><br><i>Wenbo Gong (University of Cambridge) &middot; Sebastian Tschiatschek (Microsoft Research) &middot; Sebastian Nowozin (Microsoft Research Cambridge) &middot; Richard E Turner (University of Cambridge) &middot; José Miguel Hernández-Lobato (University of Cambridge) &middot; Cheng Zhang (Microsoft)</i></p>

      <p><b>Algorithmic Guarantees for Inverse Imaging with Untrained Network Priors</b><br><i>Gauri Jagatap (Iowa State University) &middot; Chinmay Hegde (Iowa State University)</i></p>

      <p><b>Planning with Goal-Conditioned Policies</b><br><i>Soroush Nasiriany (University of California, Berkeley) &middot; Vitchyr Pong (UC Berkeley) &middot; Steven Lin (UC Berkeley) &middot; Sergey Levine (UC Berkeley)</i></p>

      <p><b>Don&#39;t take it lightly: Phasing optical random projections with unknown operators</b><br><i>Sidharth Gupta (University of Illinois at Urbana-Champaign) &middot; Remi Gribonval (INRIA) &middot; Laurent Daudet (LightOn) &middot; Ivan Dokmanic (University of Illinois at Urbana-Champaign)</i></p>

      <p><b>Generating Diverse High-Fidelity Images with VQVAE-2</b><br><i>Ali Razavi (DeepMind) &middot; Aaron van den Oord (Google Deepmind) &middot; Oriol Vinyals (Google DeepMind)</i></p>

      <p><b>Generalized Matrix Means for Semi-Supervised Learning with Multilayer Graphs</b><br><i>Pedro Mercado (University of Tübingen) &middot; Francesco Tudisco (University of Strathclyde) &middot; Matthias Hein (University of Tübingen)</i></p>

      <p><b>Online Optimal Control with Linear Dynamics and Predictions: Algorithms and Regret Analysis</b><br><i>Yingying Li (Harvard University) &middot; Xin Chen (Harvard University) &middot; Na Li (Harvard University)</i></p>

      <p><b>Missing Not at Random in Matrix Completion: The Effectiveness of Estimating Missingness Probabilities Under a Low Nuclear Norm Assumption</b><br><i>Wei Ma (Carnegie Mellon University) &middot; George Chen (Carnegie Mellon University)</i></p>

      <p><b>MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis</b><br><i>Kundan Kumar (Universite de Montreal) &middot; Rithesh Kumar (Mila) &middot; Thibault de Boissiere (Lyrebird) &middot; Lucas Gestin (Lyrebird) &middot; Wei Zhen Teoh (Lyrebird) &middot; Jose Sotelo (Lyrebird AI, MILA, Universite de Montreal) &middot; Alexandre de Brébisson (LYREBIRD, MILA) &middot; Yoshua Bengio (Mila) &middot; Aaron Courville (U. Montreal)</i></p>

      <p><b>Offline Contextual Bandits with High Probability Fairness Guarantees</b><br><i>Blossom Metevier (University of Massachusetts, Amherst) &middot; Stephen Giguere (University of Massachusetts, Amherst) &middot; Sarah Brockman (University of Massachusetts Amherst) &middot; Ari Kobren (UMass Amherst) &middot; Yuriy Brun (University of Massachusetts Amherst) &middot; Emma Brunskill (Stanford University) &middot; Philip Thomas (University of Massachusetts Amherst)</i></p>

      <p><b>Solving a Class of Non-Convex Min-Max Games Using Iterative First Order Methods</b><br><i>Maher Nouiehed (University of Southern California) &middot; Maziar Sanjabi (USC) &middot; Tianjian Huang (University of Southern California) &middot; Jason Lee (USC) &middot; Meisam Razaviyayn (University of Southern California)</i></p>

      <p><b>Semantic-Guided Multi-Attention Localization for Zero-Shot Learning</b><br><i>Yizhe Zhu (Rutgers University ) &middot; Jianwen Xie (Hikvision) &middot; Zhiqiang Tang (Rutgers) &middot; Xi Peng (University of Delaware) &middot; Ahmed Elgammal (Rutgers University)</i></p>

      <p><b>Interpreting and improving natural-language processing (in machines) with  natural language-processing (in the brain)</b><br><i>Mariya Toneva (Carnegie Mellon University) &middot; Leila Wehbe (Carnegie Mellon University)</i></p>

      <p><b>Function-Space Distributions over Kernels</b><br><i>Gregory Benton (Cornell University) &middot; Wesley J Maddox (Cornell University) &middot; Jayson Salkey (Cornell University) &middot; Julio Albinati (Microsoft) &middot; Andrew Wilson (Cornell University)</i></p>

      <p><b>SGD for Least Squares Regression: Towards Minimax Optimality with the Final Iterate</b><br><i>Rong Ge (Duke University) &middot; Sham Kakade (University of Washington) &middot; Rahul Kidambi (University of Washington) &middot; Praneeth Netrapalli (Microsoft Research)</i></p>

      <p><b>Compositional Plan Vectors</b><br><i>Coline Devin (UC Berkeley) &middot; Daniel Geng (UC Berkeley) &middot; Pieter Abbeel (UC Berkeley  Covariant) &middot; Trevor Darrell (UC Berkeley) &middot; Sergey Levine (UC Berkeley)</i></p>

      <p><b>Locally Private Learning without Interaction Requires Separation</b><br><i>Amit Daniely (Google Research) &middot; Vitaly Feldman (Google Brain)</i></p>

      <p><b>Robust Bi-Tempered Logistic Loss Based on Bregman Divergences</b><br><i>Ehsan Amid (University of California, Santa Cruz) &middot; Manfred Warmuth (Univ. of Calif. at Santa Cruz) &middot; Rohan Anil (Google) &middot; Tomer Koren (Google)</i></p>

      <p><b>Computational Separations between Sampling and Optimization</b><br><i>Kunal Talwar (Google)</i></p>

      <p><b>Surfing: Iterative Optimization Over Incrementally Trained Deep Networks</b><br><i>Ganlin Song (Yale University) &middot; Zhou Fan (Yale Univ) &middot; John Lafferty (Yale University)</i></p>

      <p><b>Population-based Meta-Optimizer Guided by Posterior Estimation</b><br><i>Yue Cao (Texas A&M University) &middot; Tianlong Chen (Texas A&M University) &middot; Zhangyang Wang (TAMU) &middot; Yang Shen (Texas A&M University)</i></p>

      <p><b>On Human-Aligned Risk Minimization</b><br><i>Liu Leqi (Carnegie Mellon University) &middot; Adarsh Prasad (Carnegie Mellon University) &middot; Pradeep Ravikumar (Carnegie Mellon University)</i></p>

      <p><b>Semi-Parametric Efficient Policy Learning with Continuous Actions</b><br><i>Victor Chernozhukov (MIT) &middot; Mert Demirer (MIT) &middot; Greg Lewis (Microsoft Research) &middot; Vasilis Syrgkanis (Microsoft Research)</i></p>

      <p><b>Multi-task Learning for Aggregated Data using Gaussian Processes</b><br><i>Fariba Yousefi (University of Sheffield) &middot; Michael Smith (University of Sheffield) &middot; Mauricio Álvarez (University of Sheffield)</i></p>

      <p><b>Minimal Variance Sampling in Stochastic Gradient Boosting</b><br><i>Bulat Ibragimov (Yandex) &middot; Gleb Gusev (Yandex)</i></p>

      <p><b>Precise and Scalable Convex Relaxations for Robustness Certification</b><br><i>Gagandeep Singh (ETH Zurich) &middot; Rupanshu Ganvir (ETH Zurich) &middot; Markus Püschel (ETH Zurich) &middot; Martin Vechev (DeepCode and ETH Zurich, Switzerland)</i></p>

      <p><b>An Algorithm to Learn Polytree Networks with Hidden Nodes</b><br><i>Firoozeh Sepehr (University of Tennessee) &middot; Donatello Materassi (University of Minnesota)</i></p>

      <p><b>Efficiently Learning Fourier Sparse Set Functions</b><br><i>Andisheh Amrollahi (ETH Zurich) &middot; Amir Zandieh (epfl) &middot; Michael Kapralov (EPFL) &middot; Andreas Krause (ETH Zurich)</i></p>

      <p><b>Projected Stein Variational Newton: A Fast and Scalable Bayesian Inference Method in High Dimensions</b><br><i>Peng Chen (The University of Texas at Austin) &middot; Keyi Wu (The University of Texas at Austin) &middot; Joshua Chen (The University of Texas at Austin) &middot; Tom O'Leary-Roseberry (The University of Texas at Austin) &middot; Omar Ghattas (The University of Texas at Austin)</i></p>

      <p><b>Invariance and identifiability issues for word embeddings</b><br><i>Rachel Carrington (University of Nottingham) &middot; Karthik Bharath (University of Nottingham) &middot; Simon Preston (University of Nottingham)</i></p>

      <p><b>Generalization Error Analysis of Quantized Compressive Learning</b><br><i>Xiaoyun Li (Rutgers University) &middot; Ping Li (Baidu Research USA)</i></p>

      <p><b>Multi-Criteria Dimensionality Reduction with Applications to Fairness</b><br><i>Uthaipon Tantipongpipat (Georgia Tech) &middot; Samira Samadi (Georgia Tech) &middot; Mohit Singh (Georgia Tech) &middot; Jamie Morgenstern (Georgia Tech) &middot; Santosh Vempala (Georgia Tech)</i></p>

      <p><b>Efficient Rematerialization for Deep Networks</b><br><i>Ravi Kumar (Google) &middot; Manish Purohit (Google) &middot; Zoya Svitkina (Google) &middot; Erik Vee (Google) &middot; Joshua Wang (Google)</i></p>

      <p><b>Fast Agent Resetting in Training</b><br><i>Samuel Ainsworth (University of Washington) &middot; Matt Barnes (University of Washington) &middot; Siddhartha Srinivasa (Amazon + University of Washington)</i></p>

      <p><b>Heterogeneous Treatment Effects with Instruments</b><br><i>Vasilis Syrgkanis (Microsoft Research) &middot; Victor Lei (Trip Advisor) &middot; Miruna Oprescu (Microsoft Research) &middot; Maggie Hei (Microsoft) &middot; Keith Battocchi (Microsoft) &middot; Greg Lewis (Microsoft Research)</i></p>

      <p><b>Understanding Sparse JL for Feature Hashing</b><br><i>Meena Jagadeesan (Harvard University)</i></p>

      <p><b>Constraint Augmented Reinforcement Learning for Text-based Recommendation and Generation</b><br><i>Ruiyi Zhang (Duke University) &middot; Tong Yu (Samsung Research America) &middot; Yilin Shen (Samsung Research America) &middot; Hongxia Jin (Samsung Research America) &middot; Changyou Chen (University at Buffalo)</i></p>

      <p><b>Flexible Modeling of Diversity with Strongly Log-Concave Distributions</b><br><i>Joshua Robinson (MIT) &middot; Suvrit Sra (MIT) &middot; Stefanie Jegelka (MIT)</i></p>

      <p><b>Momentum-Based Variance Reduction in Non-Convex SGD</b><br><i>Ashok Cutkosky (Google Research) &middot; Francesco Orabona (Boston University)</i></p>

      <p><b>Search on the Replay Buffer: Bridging Planning and Reinforcement Learning</b><br><i>Ben Eysenbach (Carnegie Mellon University) &middot; Ruslan Salakhutdinov (Carnegie Mellon University) &middot; Sergey Levine (UC Berkeley)</i></p>

      <p><b>Can Unconditional Language Models Recover Arbitrary Sentences?</b><br><i>Nishant Subramani (New York University) &middot; Samuel Bowman (New York University) &middot; Kyunghyun Cho (NYU)</i></p>

      <p><b>Group Retention when Using Machine Learning in Sequential Decision Making: the Interplay between User Dynamics and Fairness </b><br><i>Xueru Zhang (University of Michigan) &middot; Mohammad Mahdi Khalili (university of michigan) &middot; Cem Tekin (Bilkent University) &middot; mingyan liu (university of Michigan, Ann Arbor)</i></p>

      <p><b>Faster width-dependent algorithm for mixed packing and covering LPs</b><br><i>Digvijay P Boob (Georgia Institute of Technology) &middot; Saurabh Sawlani (Georgia Institute of Technology) &middot; Di Wang (Georgia Institute of Technology)</i></p>

      <p><b>Flattening a Hierarchical Clustering through Active Learning</b><br><i>Fabio Vitale (Sapienza University of Rome) &middot; Anand Rajagopalan (Google) &middot; Claudio Gentile (Google Research)</i></p>

      <p><b>DeepWave: A Recurrent Neural-Network for Real-Time Acoustic Imaging</b><br><i>Matthieu SIMEONI (IBM Research / EPFL) &middot; Sepand Kashani (EPFL) &middot; Paul Hurley (Western Sydney University) &middot; Martin Vetterli (EPFL)</i></p>

      <p><b>Certifying Geometric Robustness of Neural Networks</b><br><i>Mislav Balunovic (ETH Zurich) &middot; Maximilian Baader (ETH Zürich) &middot; Gagandeep Singh (ETH Zurich) &middot; Timon Gehr (ETH Zurich) &middot; Martin Vechev (DeepCode and ETH Zurich, Switzerland)</i></p>

      <p><b>Goal-conditioned Imitation Learning</b><br><i>Yiming Ding (University of California, Berkeley) &middot; Carlos Florensa (UC Berkeley) &middot; Pieter Abbeel (UC Berkeley  Covariant) &middot; Mariano Phielipp (Intel AI Labs)</i></p>

      <p><b>Robust exploration in linear quadratic reinforcement learning </b><br><i>Jack Umenberger (Uppsala University) &middot; Mina Ferizbegovic (KTH Royal Institute of Technology) &middot; Thomas Schön (Uppsala University) &middot; Håkan Hjalmarsson (KTH)</i></p>

      <p><b>DRUM: End-To-End Differentiable Rule Mining On Knowledge Graphs</b><br><i>Ali Sadeghian (University of Florida) &middot; Mohammadreza Armandpour (Texas A&M University) &middot; Patrick Ding (Texas A&M University) &middot; Daisy Zhe Wang (Univeresity of Florida)</i></p>

      <p><b>Kernel Truncated Randomized Ridge Regression: Optimal Rates and Low Noise Acceleration</b><br><i>Kwang-Sung Jun (Boston University) &middot; Ashok Cutkosky (Google Research) &middot; Francesco Orabona (Boston University)</i></p>

      <p><b>Input-Output Equivalence of Unitary and Contractive RNNs</b><br><i>Melikasadat Emami (UCLA) &middot; Mojtaba Sahraee Ardakan (UCLA) &middot; Sundeep Rangan (NYU) &middot; Alyson Fletcher (UCLA)</i></p>

      <p><b>Hamiltonian Neural Networks</b><br><i>Samuel Greydanus (Google Brain) &middot; Misko Dzumba (PetCube) &middot; Jason Yosinski (Uber AI Labs)</i></p>

      <p><b>Preventing Gradient Attenuation in Lipschitz Constrained Convolutional Networks</b><br><i>Qiyang Li (University of Toronto) &middot; Saminul Haque (University of Toronto) &middot; Cem Anil (University of Toronto; Vector Institute) &middot; James Lucas (University of Toronto) &middot; Roger Grosse (University of Toronto) &middot; Joern-Henrik Jacobsen (Vector Institute)</i></p>

      <p><b>Deep and Structured Similarity Matching via Deep and Structured Hebbian/Anti-Hebbian Networks</b><br><i>Dina Obeid (Harvard University) &middot; Cengiz Pehlevan (Harvard University)</i></p>

      <p><b>Understanding the Representation Power of Graph Neural Networks in Learning Graph Topology</b><br><i>Nima Dehmamy (Northeastern University) &middot; Albert-Laszlo Barabasi (Northeastern University) &middot; Rose Yu (Northeastern University)</i></p>

      <p><b>Multiple Futures Prediction</b><br><i>Charlie Tang (Apple Inc.) &middot; Ruslan Salakhutdinov (Carnegie Mellon University)</i></p>

      <p><b>Explicitly disentangling image content from translation and rotation with spatial-VAE</b><br><i>Tristan Bepler (MIT) &middot; Ellen Zhong (Massachusetts Institute of Technology) &middot; Kotaro Kelley (New York Structural Biology Center) &middot; Edward Brignole (Massachusetts Institute of Technology) &middot; Bonnie Berger (MIT)</i></p>

      <p><b>A Perspective on False Discovery Rate Control via Knockoffs</b><br><i>Jingbo Liu (MIT) &middot; Philippe Rigollet (MIT)</i></p>

      <p><b>A Kernel Loss for Solving the Bellman Equation</b><br><i>Yihao Feng (The University of Texas at Austin) &middot; Lihong Li (Google Brain) &middot; Qiang Liu (UT Austin)</i></p>

      <p><b>Low-Rank Bandit Methods for High-Dimensional Dynamic Pricing</b><br><i>Jonas Mueller (Amazon Web Services) &middot; Vasilis Syrgkanis (Microsoft Research) &middot; Matt Taddy (Chicago Booth)</i></p>

      <p><b>Differential Privacy Has Disparate Impact on Model Accuracy</b><br><i>Eugene Bagdasaryan (Cornell Tech, Cornell University) &middot; Omid Poursaeed (Cornell University) &middot; Vitaly Shmatikov (Cornell University)</i></p>

      <p><b>Riemannian batch normalization for SPD neural networks</b><br><i>Daniel Brooks (Thales) &middot; Olivier Schwander (Sorbonne Université) &middot; Frederic Barbaresco (THALES LAND & AIR SYSTEMS) &middot; Jean-Yves Schneider (THALES LAND & AIR SYSTEMS) &middot; Matthieu Cord (Sorbonne University)</i></p>

      <p><b>Neural Taskonomy: Inferring the Similarity of Task-Derived Representations from Brain Activity</b><br><i>Aria Y Wang (Carnegie Mellon University) &middot; Leila Wehbe (Carnegie Mellon University) &middot; Michael J Tarr (Carnegie Mellon University)</i></p>

      <p><b>Stacked Capsule Autoencoders</b><br><i>Adam Kosiorek (University of Oxford) &middot; Sara Sabour (Google) &middot; Yee Whye Teh (University of Oxford, DeepMind) &middot; Geoffrey E Hinton (Google & University of Toronto)</i></p>

      <p><b>Learning Reward Machines for Partially Observable Reinforcement Learning</b><br><i>Rodrigo Toro Icarte (University of Toronto and Vector Institute) &middot; Ethan Waldie (University of Toronto) &middot; Toryn Klassen (University of Toronto) &middot; Rick Valenzano (Element AI) &middot; Margarita Castro (University of Toronto) &middot; Sheila McIlraith (University of Toronto)</i></p>

      <p><b>Learning Representations by Maximizing Mutual Information Across Views</b><br><i>Philip Bachman (Microsoft Research) &middot; R Devon Hjelm (Microsoft Research) &middot; William Buchwalter (Microsoft)</i></p>

      <p><b>Learning Deep MRFs with Amortized Bethe Free Energy Minimization</b><br><i>Sam Wiseman (TTIC) &middot; Yoon Kim (Harvard University)</i></p>

      <p><b>Small ReLU networks are powerful memorizers: a tight analysis of memorization capacity</b><br><i>Chulhee Yun (Massachusetts Institute of Technology) &middot; Suvrit Sra (MIT) &middot; Ali Jadbabaie (MIT)</i></p>

      <p><b>Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks</b><br><i>Aaron Voelker (University of Waterloo) &middot; Ivana Kajić (University of Waterloo) &middot; Chris Eliasmith (U of Waterloo)</i></p>

      <p><b>Exact Combinatorial Optimization with Graph Convolutional Neural Networks</b><br><i>Maxime Gasse (Polytechnique Montréal) &middot; Didier Chetelat (Polytechnique Montreal) &middot; Nicola Ferroni (University of Bologna) &middot; Laurent Charlin (MILA / U.Montreal) &middot; Andrea Lodi (École Polytechnique Montréal)</i></p>

      <p><b>Fast structure learning with modular regularization</b><br><i>Greg Ver Steeg (University of Southern California) &middot; Hrayr  Harutyunyan (USC Information Sciences Institute) &middot; Daniel Moyer (USC Information Sciences Institute) &middot; Aram Galstyan (USC Information Sciences Inst)</i></p>

      <p><b>Wasserstein Dependency Measure for Representation Learning</b><br><i>Sherjil Ozair (Université de Montréal) &middot; Corey Lynch (Google Brain) &middot; Yoshua Bengio (Mila) &middot; Aaron van den Oord (Google Deepmind) &middot; Sergey Levine (UC Berkeley) &middot; Pierre Sermanet (Google Brain)</i></p>

      <p><b>TAB-VCR: Tags and Attributes for Visual Commonsense Reasoning</b><br><i>Jingxiang Lin (University of illinois at urbana-champaign) &middot; Unnat Jain (University of Illinois at Urbana Champaign) &middot; Alexander Schwing (University of Illinois at Urbana-Champaign)</i></p>

      <p><b>Universality and individuality in neural dynamics across large populations of recurrent networks</b><br><i>Niru Maheswaranathan (Google Brain) &middot; Alex H Williams (Stanford University) &middot; Matthew Golub (Stanford University) &middot; Surya Ganguli (Stanford) &middot; David Sussillo (Google Inc.)</i></p>

      <p><b>End-to-End Learning on 3D Protein Structure for Interface Prediction</b><br><i>Raphael Townshend (Stanford University) &middot; Patricia Suriana (Stanford) &middot; Rishi Bedi (Stanford University) &middot; Ron Dror (Stanford University)</i></p>

      <p><b>A Family of Robust Stochastic Operators for Reinforcement Learning</b><br><i>Yingdong Lu (IBM Research) &middot; Mark Squillante (IBM Research) &middot; Chai Wah Wu (IBM)</i></p>

      <p><b>Improving Model Robustness and Uncertainty Estimates with Self-Supervised Learning</b><br><i>Dan Hendrycks (UC Berkeley) &middot; Mantas Mazeika (University of Chicago) &middot; Saurav Kadavath (UC Berkeley) &middot; Dawn Song (UC Berkeley)</i></p>

      <p><b>Inherent Tradeoffs in Learning Fair Representation</b><br><i>Han Zhao (Carnegie Mellon University) &middot; Geoff Gordon (Microsoft)</i></p>

      <p><b>Are deep ResNets provably better than linear predictors?</b><br><i>Chulhee Yun (Massachusetts Institute of Technology) &middot; Suvrit Sra (MIT) &middot; Ali Jadbabaie (MIT)</i></p>

      <p><b>Reverse engineering recurrent networks for sentiment classification reveals line attractor dynamics</b><br><i>Niru Maheswaranathan (Google Brain) &middot; Alex H Williams (Stanford University) &middot; Matthew Golub (Stanford University) &middot; Surya Ganguli (Stanford) &middot; David Sussillo (Google Inc.)</i></p>

      <p><b>BehaveNet: nonlinear embedding and Bayesian neural decoding of behavioral videos</b><br><i>Eleanor Batty (Columbia University) &middot; Matthew Whiteway (Columbia University) &middot; Shreya Saxena (Columbia University) &middot; Dan Biderman (Columbia University) &middot; Taiga Abe (Columbia University) &middot; Simon Musall (Cold Spring Harbor Laboratory) &middot; Winthrop Gillis (Harvard Medical School) &middot; Jeffrey Markowitz (Harvard Medical School) &middot; Anne Churchland (Cold Spring Harbor Laboratory) &middot; John Cunningham (University of Columbia) &middot; Sandeep R Datta (Harvard Medical School) &middot; Scott Linderman (Stanford University) &middot; Liam Paninski (Columbia University)</i></p>

      <p><b>Variational Mixture-of-Experts Autoencoders for Multi-Modal Deep Generative Models</b><br><i>Yuge Shi (University of Oxford) &middot; Siddharth Narayanaswamy (Unversity of Oxford) &middot; Brooks Paige (Alan Turing Institute) &middot; Philip  Torr (University of Oxford)</i></p>

      <p><b>Gradient-based Adaptive Markov Chain Monte Carlo</b><br><i>Michalis Titsias (DeepMind) &middot; Petros Dellaportas (University College London, Athens University of Economics and Alan Turing Institute)</i></p>

      <p><b>On the Role of Inductive Bias From Simulation and the Transfer to the Real World: a new Disentanglement Dataset</b><br><i>Muhammad Waleed Gondal (Max Planck Institute for Intelligent Systems) &middot; Manuel Wuthrich (Max Planck Institute for Intelligent Systems) &middot; Djordje Miladinovic (ETH Zurich) &middot; Francesco Locatello (ETH Zürich - MPI Tübingen) &middot; Martin  Breidt (MPI for Biological Cybernetics) &middot; Valentin Volchkov (Max Planck Institut for Intelligent Systems) &middot; Joel Akpo (Max Planck Institute for Intelligent Systems) &middot; Olivier Bachem (Google Brain) &middot; Bernhard Schölkopf (MPI for Intelligent Systems) &middot; Stefan Bauer (MPI for Intelligent Systems)</i></p>

      <p><b>Imitation-Projected Policy Gradient for Programmatic Reinforcement Learning</b><br><i>Abhinav Verma (Rice University) &middot; Hoang Le (California Institute of Technology) &middot; Yisong Yue (Caltech) &middot; Swarat Chaudhuri (Rice University)</i></p>

      <p><b>Learning Data Manipulation for Augmentation and Weighting</b><br><i>Zhiting Hu (Carnegie Mellon University) &middot; Bowen Tan (CMU) &middot; Ruslan Salakhutdinov (Carnegie Mellon University) &middot; Tom Mitchell (Carnegie Mellon University) &middot; Eric Xing (Petuum Inc. /  Carnegie Mellon University)</i></p>

      <p><b>Exploring Algorithmic Fairness in Robust Graph Covering Problems</b><br><i>Aida Rahmattalabi (University of Southern California) &middot; Phebe Vayanos (University of Southern California) &middot; Anthony Fulginiti (University of Denver) &middot; Eric Rice (University of Southern California) &middot; Bryan Wilder () &middot; Amulya Yadav (Pennsylvania State University) &middot; Milind Tambe (USC)</i></p>

      <p><b>Abstraction based Output Range Analysis for Neural Networks</b><br><i>Pavithra  Prabhakar (Kansas State University) &middot; Zahra Rahimi Afzal (Kansas State University)</i></p>

      <p><b>Space and Time Efficient Kernel Density Estimation in High Dimensions</b><br><i>Arturs Backurs (MIT) &middot; Piotr Indyk (MIT) &middot; Tal Wagner (MIT)</i></p>

      <p><b>PIDForest: Anomaly Detection and Certification via Partial Identification</b><br><i>Parikshit Gopalan (VMware Research) &middot; Vatsal Sharan (Stanford University) &middot; Udi Wieder (VMware Research)</i></p>

      <p><b>Generative Models for Graph-Based Protein Design</b><br><i>John Ingraham (MIT) &middot; Vikas Garg (MIT) &middot; Regina Barzilay (Massachusetts Institute of Technology) &middot; Tommi Jaakkola (MIT)</i></p>

      <p><b>The Geometry of Deep Networks: Power Diagram Subdivision</b><br><i>Randall Balestriero (Ecole Normale Superieure, Paris) &middot; Romain Cosentino (Rice University) &middot; Behnaam Aazhang (Rice University) &middot; Richard Baraniuk (Rice University)</i></p>

      <p><b>Approximate Feature Collisions in Neural Nets</b><br><i>Ke Li (UC Berkeley) &middot; Tianhao Zhang (Nanjing University) &middot; Jitendra Malik (University of California at Berkley)</i></p>

      <p><b>Ease-of-Teaching and Language Structure from Emergent Communication</b><br><i>Fushan Li (University of Alberta) &middot; Michael Bowling (University of Alberta)</i></p>

      <p><b>Generalization in multitask deep neural classifiers: a statistical physics approach</b><br><i>Anthony Ndirango (Intel AI Lab) &middot; Tyler Lee (Intel AI Lab)</i></p>

      <p><b>Distributionally Optimistic Optimization Approach to Nonparametric Likelihood Approximation</b><br><i>Viet Anh Nguyen (EPFL) &middot; Soroosh Shafieezadeh Abadeh (EPFL) &middot; Man-Chung Yue (The Hong Kong Polytechnic University) &middot; Daniel Kuhn (EPFL) &middot; Wolfram Wiesemann (Imperial College)</i></p>

      <p><b>On Relating Explanations and Adversarial Examples</b><br><i>Alexey Ignatiev (Reason Lab, Faculty of Sciences, University of Lisbon) &middot; Nina Narodytska (VMWare Research) &middot; Joao Marques-Silva (Reason Lab, Faculty of Sciences, University of Lisbon)</i></p>

      <p><b>On the equivalence between graph isomorphism testing and function approximation with GNNs</b><br><i>Zhengdao Chen (New York University) &middot; Soledad Villar (New York University) &middot; Lei Chen (New York University) &middot; Joan Bruna (NYU)</i></p>

      <p><b>Surround Modulation: A Bio-inspired Connectivity Structure for Convolutional Neural Networks</b><br><i>Hosein Hasani (Sharif University of Technology) &middot; Mahdieh Soleymani (Sharif University of Technology) &middot; Hamid Aghajan (Sharif University of Technology and iMinds, Gent University,)</i></p>

      <p><b>Self-attention with Functional Time Representation Learning</b><br><i>Da Xu (Walmart Labs) &middot; Chuanwei Ruan (Walmart Labs) &middot; Evren Korpeoglu (Walmart Labs) &middot; Sushant Kumar (Walmart Labs) &middot; Kannan Achan (Walmart Labs)</i></p>

      <p><b>Re-randomized Densification for One Permutation Hashing and Bin-wise Consistent Weighted Sampling</b><br><i>Ping Li (Baidu Research USA) &middot; xiaoyun Li (Rutgers) &middot; Cun-Hui Zhang (Rutgers)</i></p>

      <p><b>Enabling hyperparameter optimization in sequential autoencoders for spiking neural data</b><br><i>Mohammad Reza Keshtkaran (Emory University and Georgia Tech) &middot; Chethan Pandarinath (Emory University and Georgia Tech)</i></p>"""


def write_textfile(filepath, lines, append=False, with_newline=True):
    mode = 'a' if append else 'w'

    with open(filepath, mode) as f:
        for line in lines:
            f.write(line)
            if with_newline:
                f.write("\n")


lst = [x.strip() for x in s.split('\n') if "Stanford" in x]
print(len(lst))
write_textfile('stanford.txt', lst)
lst = [
    x.strip()
    for x in s.split('\n')
    if "MIT" in x or "Massachusetts Institute of Technology" in x
]
print(len(lst))
write_textfile('mit.txt', lst)
lst = [
    x.strip()
    for x in s.split('\n')
    if "CMU" in x or "Carnegie Mellon University" in x
]
print(len(lst))
write_textfile('cmu.txt', lst)
lst = [x.strip() for x in s.split('\n') if "Berkeley" in x]
print(len(lst))
write_textfile('berkeley.txt', lst)