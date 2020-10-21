clc;clear;clear all;

[interaction] =importdata(["C:\Users\deept\Desktop\cRNA-Gkl\Dataset-3\chr_diseasematrix.csv"]);

  %[ndr,nm] =  size(interaction);
  [nc,nd ]= size(interaction);
  
 [SS] =importdata(["C:\Users\deept\Desktop\cRNA-Gkl\Dataset-3\dissimilarity.csv"]);
 [CS] =importdata(["C:\Users\deept\Desktop\cRNA-Gkl\Dataset-3\seqsimilarity.csv"]);
 
 [CC,DD]=gkl(nc,nd,interaction);
 %[kd,km,X2] = sigmoid(interaction,nc,nd);
 
%  circ_feature=PCA((circsim+CC)/2,a); %circ
%  dis_feature=PCA((dissim+DD)/2,b);   %dd

  circ_feature= (CS+CC)/2;   %circ
  dis_feature= (SS+DD)/2;    %dd
 
%  a= 0.6; b= 0.9;
% circ_feature=PCA((SS+CC)/2,a);   %circ
% dis_feature=PCA((CS+DD)/2,b);    %dd

  dlmwrite('C:\Users\deept\Desktop\integrated-cRNA similarity.txt',circ_feature);
  dlmwrite('C:\Users\deept\Desktop\integrated-disease similarity.txt',dis_feature);