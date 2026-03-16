
clear;clc;
n=2;
L=0.5;
tau1=0.1;tau2=0.2;tau12=tau2-tau1;
rho1=0.3;rho2=0.3;rho3=0.3;rho4=0.3;
eta=0.1;



a=6;
e=@(i) [zeros(n,(i-1)*n),eye(n,n),zeros(n,(22-i)*n)];
e0=zeros(n,22*n);
phi1_1=[e(1);e(2);e(6);e(21)];
phi1_2=[e(2);e(3);e(8);e(22)];
phi2_1 = [e(1); e(2)];
phi2_2 = [e(5); e(7)];
phi2_3 = [e(6); e(8)];
phi2_4 = [e(2); e(3)];
phi2_5 = [e(21); e(22)];


omg1_1 = [e(1) - e(9); e(6) - e(11)];
omg2_1 = [e(10) - e(1); e(12) - e(6)];
omg1_2 = [e(2); e(8)];
omg2_2 = -omg1_2;
omg4_1 = [omg1_1; omg2_1];
omg5 = [e(9); e(10); e(11); e(12)];
omg4_2 = [omg1_2; omg2_2];
omg6_1=[omg5;e(15);e(16)];
omg6_2=[e0;e0;e0;e0;e(1)-e(15);e0];
omg6_3=[e0;e0;e0;e0;e0;e(16)-e(1)];
omg3_3=[omg1_2;e0;e0];
omg3_4=[e0;e0;omg2_2];
omg3_5=[-omg1_1;omg2_1];
omg3_1=[omg1_1;e0;e0];
omg3_2=[e0;e0;omg2_1];

phi0=[e(1)-e(4);e(4)-e(5)];
phi0_1=e(1)-e(6);
phi0_2=e(1)+e(6)-2*e(17);

F1=e(1)-e(9);
F01=e(1)+e(9);
F2=F01-2*e(15);
F02=e(10)+e(1);
F12=[F1;F2];
F3=e(10)-e(1);
F4=F02-2*e(16);
F34=[F3;F4];
F56=[F1;3*F2];
E1=[e(1);e(15)];
E2=[e(4);e(16)];

W1=e(1)-e(18);
W2=e(1)+e(18)-2*e(19);
W3=[e(4);e(5);e(18)];

GAMA0=e(1)'+rho1*e(2)'+rho2*e(3)'+rho3*e(9)';%没改
% alpha1=[1.9,0;0,1.9];
% beta1=[2.3,0;0,2.3];
% ksai1=[0.5,0.6;1.5,1.4];ksai2=[1,0.2;1,2.5];
% gama1=[1.5,1;4,1.1]; gama2=[1,2.1;1.1,-1];
% ga1=[0.1,0.35;0.3,1.1];ga2=[0.1,1;1.1,-1];

alpha1=[0.9,0;0,1.2];
beta1=[2.3,0;0,2.3];%%%%%%%%%%%%%%%%
ksai1=[-0.5+1i,0.6+0.2i;1.5-1i,1.4+2.5i];
gama1=[-1.5+3.1i,1+2.1i;4+1.1i,1.1-1i];%忆阻部分
gama2=[-0.1+0.1i,0.35+1i;0.3+1.1i,1.1-1i];%忆阻部分

 
%%uncertain
II2=[1,0;0,1];
Hab1=II2;Hab2=II2;
Na1=0.2*II2;Na2=0.2*II2;Nb1=0.2*II2;Nb2=0.2*II2;
Hkr1=II2;Hkr2=II2;
Nk1=0.3*II2;Nk2=0.3*II2;Nr1=0.3*II2;Nr2=0.3*II2;
%%控制器中uncertain
H1=II2;H2=II2;
N1=0.5*II2;N2=0.5*II2;Ntau1=0.5*II2;Ntau2=0.5*II2;

% delta1=[0.12+0.1i,0.3-0.2i;0.2+1i,-0.1+0.3i];
% % delta2=[0.2-0.1i,-0.1+0.2i;0.1-1i,0.2-1i];%%控制器参数%%%%%%%%%%%%%%

%%
GAMA=[0.5,0;0,0.5];%李普希茨条件
%%
iteration = 0;  % 记录迭代次数
disp('开始迭代过程：');
 h1=0.013;h2=0.376;tol=0.001;
while (h2-h1)>tol
    %i=i+1;
    iteration = iteration + 1;
    h=(h1+h2)/2;

setlmis([]);

% 定义变量 (复数 Hermitian 矩阵)
    P1 = sdpvar(8, 8, 'hermitian', 'complex');
    Q11 = sdpvar(2, 2, 'hermitian', 'complex');
    Q21 = sdpvar(4, 4, 'hermitian', 'complex');
    Q31 = sdpvar(2, 2,'hermitian', 'complex'); 
      
    Y1=sdpvar(4, 4, 'full', 'complex');
    Y2=sdpvar(4, 4, 'full', 'complex');
    % 定义对称矩阵 O1
    X = sdpvar(4, 4, 'full', 'complex');
    S1 = sdpvar(size(omg4_1, 1), size(omg4_1, 1), 'full', 'complex');
    S2 = sdpvar(size(omg3_5, 1), size(omg3_5, 1), 'full', 'complex');
    G=sdpvar(2, 2, 'full', 'complex');
    % 定义变量 (使用 YALMIP 定义)
    Y = sdpvar(12, 12, 'hermitian', 'complex'); 
    R2 = sdpvar(4, 4, 'hermitian', 'complex');
    R3 = sdpvar(2, 2, 'hermitian', 'complex');
    M1 = sdpvar(2, 2, 'hermitian', 'complex');
     M2 = sdpvar(4, 4, 'hermitian', 'complex');
    KK1=sdpvar(2, 2, 'full', 'complex');
    KKtau1=sdpvar(2, 2, 'full', 'complex');
  
    mu = 2;  % 定义标量 mu
    ee1 = 2; % 定义标量 ee1
    ee2 = 2; % 定义标量 ee2
    
 D1 = sdpvar(1, 1);
D2 = sdpvar(1, 1);
D3 = sdpvar(1, 1);
D4 = sdpvar(1, 1);
DELTA1=diag([D1, D2]);
DELTA2=diag([D3, D4]);%实对角
 


    O1 = sdpvar(2, 2, 'hermitian', 'complex');
    OO1 = [O1, zeros(2); zeros(2), 3 * O1]; % 4x4 复合矩阵

    O2 = sdpvar(2, 2, 'hermitian', 'complex');
    OO2 = [O2, zeros(2); zeros(2), 3 * O2]; % 4x4 复合矩阵

    
    OOO1 = [O1, zeros(2); zeros(2), 1/3 * O1];
    OOO2 = [O2, zeros(2); zeros(2), 1/3 * O2];

GAMA1=sdpvar(2,2,'full','complex');       % 定义一般矩阵变量 GAMA1
QQ11 = [ -2*Q11 + GAMA1 + GAMA1', Q11 - GAMA1, Q11 - GAMA1';
         Q11 - GAMA1',            -Q11,         GAMA1';
         Q11 - GAMA1,             GAMA1,         -Q11];
     
    term1=phi1_1' * P1 * phi1_2+omg3_5' * S1 * omg4_1 + omg3_5' * S2 * omg5+2*a*omg1_1'*X*omg2_1+omg1_2'*X*omg2_1+omg1_1'*X*omg2_2-GAMA0*G'*e(3)-GAMA0*alpha1*G'*e(2)-GAMA0*beta1*G'*e(1)+GAMA0*ksai1*G'*e(13)+GAMA0*gama1*G'*e(14)+GAMA0*gama2*G'*e(20)+GAMA0*KK1*e(9)+GAMA0* KKtau1*e(11);
    term2=2*a*phi1_1'*P1*phi1_1+tau1^2*e(2)'* Q31*e(2) + phi2_1'*Q21*phi2_1+tau12^2*e(2)'*Q11*e(2)-exp(-2*a*tau2)*phi2_2'*Q21*phi2_2-exp(-2*a*tau1)*W1'*Q31*W1-3*exp(-2*a*tau1)*W2'*Q31*W2+exp(-2*a*tau2)*W3'*QQ11*W3;
    term3=L^2*e(13)'*M1*e(13)-exp(-2*a*L)*e(20)'*M1*e(20)+phi2_1'*M2*phi2_1-exp(-2*a*L)*phi2_5'*M2*phi2_5;
    term4=2*a*h^2*omg6_1'*Y*omg6_1+phi2_1'*R2*phi2_1-exp(-2*a*eta)*phi2_3'*R2*phi2_3+eta^2*e(2)'*R3*e(2)-exp(-2*a*eta)*phi0_1'*R3*phi0_1-exp(-2*a*eta)*3*phi0_2'*R3*phi0_2-exp(-2*a*h)*F12'* OO1*F12-exp(-2*a*h)*F34'*OO2*F34;
   
    
    term6=-exp(-2*a*h)*F12'*OO1*F12+h*omg6_1'*Y*omg6_1+h^2*e(2)'*O1*e(2);
    term7=h*2*a*omg3_1'*S1*omg4_1+h*2*a*omg3_1'*S2*omg5+h*omg3_3'*S1*omg4_1+h*omg3_3'*S2*omg5+h*omg3_1'*S1*omg4_2+h*omg6_2'*Y*omg6_1-exp(-2*a*h)*F56'*Y2*F34+...
         (h*2*a*omg3_1'*S1*omg4_1+h*2*a*omg3_1'*S2*omg5+h*omg3_3'*S1*omg4_1+h*omg3_3'*S2*omg5+h*omg3_1'*S1*omg4_2+h*omg6_2'*Y*omg6_1-exp(-2*a*h)*F56'*Y2*F34)';
    
    
    term77=h*2*a*omg3_2'*S1*omg4_1+h*2*a*omg3_2'*S2*omg5+h*omg3_4'*S1*omg4_1+h*omg3_4'*S2*omg5+h*omg3_2'*S1*omg4_2+h*omg6_3'*Y*omg6_1-exp(-2*a*h)*F56'*Y1*F34+...
          (h*2*a*omg3_2'*S1*omg4_1+h*2*a*omg3_2'*S2*omg5+h*omg3_4'*S1*omg4_1+h*omg3_4'*S2*omg5+h*omg3_2'*S1*omg4_2+h*omg6_3'*Y*omg6_1-exp(-2*a*h)*F56'*Y1*F34)';
    term66=-exp(-2*a*h)*F34'*OO2*F34-h*omg6_1'*Y*omg6_1+h^2*e(2)'*O2*e(2);
   
    term8=-e(13)' * DELTA1 * e(13) + e(1)' * GAMA^2 * DELTA1 * e(1) -e(14)' * DELTA2 * e(14) + e(4)' * GAMA^2 * DELTA2 * e(4) ;

    T1= [term1+(term1)'+term2+term3+term4+term8,                         GAMA0*H1,      GAMA0*Hab1,        GAMA0*Hkr1,         ee1 * (e(2)'*G*Na1'+ e(1)'*G*Nb1'),        ee2 * (e(13)' * G * Nk1'+ e(14)' * G * Nr1'+e(20)' * G * Nr1'), mu * (e(9)' * G * N1'+ e(11)' * G * Ntau1') ;
         (GAMA0*H1)',                                               -mu*eye(n),         zeros(n),           zeros(n),              zeros(n),                                              zeros(n),                                                   zeros(n);
         (GAMA0*Hab1)',                                                     zeros(n),           -ee1*eye(n),        zeros(n),             zeros(n),                                              zeros(n),                                                   zeros(n);
         (GAMA0*Hkr1)',                                                     zeros(n),           zeros(n),          -ee2*eye(n) ,           zeros(n),                                             zeros(n),                                                   zeros(n);
( ee1 * (e(2)'*G*Na1'+ e(1)'*G*Nb1'))',                                     zeros(n),         zeros(n),           zeros(n)  ,             -ee1*eye(n),                                           zeros(n),                                                   zeros(n);
 (ee2 * (e(13)' * G * Nk1'+ e(14)' * G * Nr1'+e(20)' * G * Nr1'))',         zeros(n),         zeros(n),           zeros(n) ,               zeros(n),                                            -ee2*eye(n),                                                 zeros(n);
 (mu * (e(9)' * G * N1'+ e(11)' * G * Ntau1'))',                            zeros(n),         zeros(n),           zeros(n)  ,              zeros(n),                                             zeros(n),                                                  -mu*eye(n)];

T2=[term1+(term1)'+term2+term3+term4+term6+term7+term8,               GAMA0*H1,     GAMA0*Hab1,   GAMA0*Hkr1,       F12'*Y1,            ee1 * (e(2)'*G*Na1'+ e(1)'*G*Nb1'),ee2 * (e(13)' * G * Nk1'+ e(14)' * G * Nr1'+e(20)' * G * Nr1'), mu * (e(9)' * G * N1'+ e(11)' * G * Ntau1') ;
(GAMA0*H1)',                                                      -mu*eye(n),       zeros(n),     zeros(n),         zeros(n, 2*n),                  zeros(n),                           zeros(n),                                        zeros(n);
(GAMA0*Hab1)',                                                            zeros(n),        -ee1*eye(n),  zeros(n),         zeros(n, 2*n),                  zeros(n),                           zeros(n),                                        zeros(n);
(GAMA0*Hkr1)',                                                            zeros(n),         zeros(n),    -ee2*eye(n),      zeros(n, 2*n),                  zeros(n),                           zeros(n),                                        zeros(n);
( F12'*Y1)',                                                            zeros(2*n, n),    zeros(2*n,n),zeros(2*n, n),  -exp(2*a*h)*OOO2,                  zeros(2*n, n),                     zeros(2*n, n),                                  zeros(2*n, n);
( ee1 * (e(2)'*G*Na1'+ e(1)'*G*Nb1'))',                                    zeros(n),         zeros(n),      zeros(n)  ,         zeros(n, 2*n),            -ee1*eye(n),                      zeros(n),                                       zeros(n);
 (ee2 * (e(13)' * G * Nk1'+ e(14)' * G * Nr1'+e(20)' * G * Nr1'))',        zeros(n),         zeros(n),      zeros(n)   ,        zeros(n, 2*n),             zeros(n),                      -ee2*eye(n),                                     zeros(n);
 (mu * (e(9)' * G * N1'+ e(11)' * G * Ntau1'))',                           zeros(n),         zeros(n),      zeros(n)  ,         zeros(n, 2*n),             zeros(n),                      zeros(n),                                        -mu*eye(n)];

T3=[term1+(term1)'+term2+term3+term4+term66+term77+term8,    GAMA0*H1,    GAMA0*Hab1,         GAMA0*Hkr1,             F34'*Y2',        ee1 * (e(2)'*G*Na1'+ e(1)'*G*Nb1'),ee2 * (e(13)' * G * Nk1'+ e(14)' * G * Nr1'+e(20)' * G * Nr1'), mu * (e(9)' * G * N1'+ e(11)' * G * Ntau1') ;
(GAMA0*H1)',                                           -mu*eye(n),          zeros(n),           zeros(n),           zeros(n, 2*n),                zeros(n),                           zeros(n),                                        zeros(n);
(GAMA0*Hab1)',                                                zeros(n),            -ee1*eye(n),        zeros(n),           zeros(n, 2*n),                 zeros(n),                           zeros(n),                                        zeros(n);
(GAMA0*Hkr1)',                                                zeros(n),             zeros(n),          -ee2*eye(n),        zeros(n, 2*n),                 zeros(n),                           zeros(n),                                        zeros(n);
( F34'*Y2')',                                                  zeros(2*n, n),        zeros(2*n, n),     zeros(2*n, n),     -exp(2*a*h)*OOO1,           zeros(2*n, n),                     zeros(2*n, n),                                  zeros(2*n, n);
( ee1 * (e(2)'*G*Na1'+ e(1)'*G*Nb1'))',                       zeros(n),           zeros(n),            zeros(n)  ,         zeros(n, 2*n),            -ee1*eye(n),                      zeros(n),                                       zeros(n);
 (ee2 * (e(13)' * G * Nk1'+ e(14)' * G * Nr1'+e(20)' * G * Nr1'))',             zeros(n),           zeros(n),            zeros(n)  ,         zeros(n, 2*n),               zeros(n),                      -ee2*eye(n),                                     zeros(n);
 (mu * (e(9)' * G * N1'+ e(11)' * G * Ntau1'))',              zeros(n),           zeros(n),             zeros(n)  ,         zeros(n, 2*n),               zeros(n),                      zeros(n),                                        -mu*eye(n)];

 T4=[Q11,GAMA1;GAMA1',Q11];
  
    % 引入一个小的正数 ε
    epsilon = 1e-6; % 可以根据需要调整这个值
  Constraints = [T1 + epsilon * eye(size(T1)) <= 0,...
                 T2 + epsilon * eye(size(T2)) <= 0,...
                 T3 + epsilon * eye(size(T3)) <= 0,...  
                 T4 >= epsilon * eye(size(T4)),...  
                 P1 >= epsilon * eye(size(P1)),...
                  M1 >= epsilon * eye(size(M1)),...
                   M2 >= epsilon * eye(size(M2)),...
                 Q11>= epsilon * eye(size(Q11)),...
                 Q21 >= epsilon * eye(size(Q21)),...
                  Q31>= epsilon * eye(size(Q31)),...
                 R2>= epsilon * eye(size(R2)),...
                 R3>= epsilon * eye(size(R3)),...
                O1 >= epsilon * eye(size(O1)),...
                O2 >= epsilon * eye(size(O2)),...
                  DELTA1 >= epsilon * eye(size(DELTA1));...
    DELTA2 >= epsilon * eye(size(DELTA2))];
    % 求解
   options = sdpsettings('solver', 'mosek', 'verbose', 0);
    diagnostics = optimize(Constraints, [], options);
    disp(['第 ', num2str(iteration), ' 次迭代:']);
    disp(['当前 h = ', num2str(h)]);
     if diagnostics.problem == 0
        % 解可行，更新下界 h1
        h1 = h;
        disp('  优化求解成功，解可行！');
DELTA1_value=value(DELTA1);
DELTA2_value=value(DELTA2);
P1_value=value(P1);
KK1_value = value(KK1);  % KK1的具体数值
KKtau1_value = value(KKtau1);  % KKtau1 的具体数值
  
G_value = value(G);   % 获取 G 的数值
G_inv_H = inv(G_value'); % (G^H)^(-1) 计算共轭转置的逆



% 计算 KKK1 = KK1 * G_inv_transpose
KKK1 = KK1_value* G_inv_H ;
KKKtau1 = KKtau1_value*G_inv_H ;
    % 显示 KK1_r, KK1_i 和 KK1
    fprintf('DELTA1 的值为:\n');
    disp(DELTA1_value);
%     
%     fprintf('P1 的值为:\n');
%     disp(P1_value);
%    
  
    fprintf('KK1 的值为:\n');
    disp(KK1_value);


    fprintf('KKtau1 的值为:\n');
    disp(KKtau1_value);
  
    fprintf('G 的值为:\n');
    disp(G_value);
 fprintf('KKK1 的值为:\n');
    disp(KKK1);
    fprintf('KKKtau1 的值为:\n');
    disp(KKKtau1);
    else
        % 解不可行，更新上界 h2
        h2 = h;
        disp('  优化求解失败，解不可行。');
        disp(['  diagnostics.problem = ', num2str(diagnostics.problem)]);
        disp(diagnostics.info);  % 输出更多诊断信息
    end
end

% 最终结果输出
disp('迭代结束！');
disp(['最终收敛的 h = ', num2str(h1)]);


