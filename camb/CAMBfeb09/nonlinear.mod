  �%  _   k820309    �          2021.9.0    (�?d                                                                                                          
       halofit.f90 NONLINEAR              MIN_KH_NONLINEAR NONLINEAR_GETNONLINRATIOS                                                     
                                                           
                     @  @                                '�                   #NUM_K    #NUM_Z    #LOG_KH    #REDSHIFTS    #MATPOWER    #DDMAT 	   #NONLIN_RATIO 
                �                                                               �                                                             �                                                          
            &                                                       �                                         P                 
            &                                                       �                                         �                 
            &                   &                                                       �                             	            �                 
            &                   &                                                       �                             
            X                
            &                   &                                                          @  @                                '�             .      #WANTCLS    #WANTTRANSFER    #WANTSCALARS    #WANTTENSORS    #WANTVECTORS    #DOLENSING    #NONLINEAR    #MAX_L    #MAX_L_TENSOR    #MAX_ETA_K    #MAX_ETA_K_TENSOR    #OMEGAB    #OMEGAC    #OMEGAV    #OMEGAN    #H0    #TCMB    #YHE    #NUM_NU_MASSLESS    #NUM_NU_MASSIVE    #NU_MASS_SPLITTINGS     #NU_MASS_EIGENSTATES !   #NU_MASS_DEGENERACIES "   #NU_MASS_FRACTIONS #   #SCALAR_INITIAL_CONDITION $   #OUTPUTNORMALIZATION %   #ACCURATEPOLARIZATION &   #ACCURATEBB '   #ACCURATEREIONIZATION (   #MASSIVENUMETHOD )   #INITPOWER *   #REION 4   #RECOMB <   #TRANSFER A   #INITIALCONDITIONVECTOR H   #ONLYTRANSFERS I   #REIONHIST J   #FLAT R   #CLOSED S   #OPEN T   #OMEGAK U   #CURV V   #R W   #KSIGN X   #TAU0 Y   #CHI0 Z                �                                                               �                                                              �                                                              �                                                              �                                                              �                                                              �                                                              �                                                              �                                            	                   �                                   (       
   
                �                                   0          
                �                                   8          
                �                                   @          
                �                                   H          
                �                                   P          
                �                                   X          
                �                                   `          
                �                                   h          
                �                                   p          
                �                                   x          
                �                                     �                          �                               !     �                          �                              "            �                 
  p          p            p                                       �                              #            �                 
  p          p            p                                       �                               $     �                          �                               %     �                          �                               &     �                          �                               '     �                          �                               (     �                          �                               )     �                          �                               *     �       �              #INITIALPOWERPARAMS +                  @  @                          +     '�                    #NN ,   #AN -   #N_RUN .   #ANT /   #RAT 0   #K_0_SCALAR 1   #K_0_TENSOR 2   #SCALARPOWERAMP 3                � $                              ,                                � $                             -                             
  p          p            p                                       � $                             .            0                 
  p          p            p                                       � $                             /            X                 
  p          p            p                                       � $                             0            �                 
  p          p            p                                       � $                             1     �          
                � $                             2     �          
                � $                             3            �                 
  p          p            p                                       �                               4     (       �              #REIONIZATIONPARAMS 5                  @  @                          5     '(                    #REIONIZATION 6   #USE_OPTICAL_DEPTH 7   #REDSHIFT 8   #DELTA_REDSHIFT 9   #FRACTION :   #OPTICAL_DEPTH ;                �                               6                                �                               7                               �                              8               
                �                              9               
                �                              :               
                �                              ;                
                �                               <            �      !       #RECOMBINATIONPARAMS =                  @  @                          =     '                    #RECFAST_FUDGE >   #RECFAST_FUDGE_HE ?   #RECFAST_HESWITCH @                � $                             >                
                � $                             ?               
                � $                              @                               �                               A                 "       #TRANSFERPARAMS B                  @  @                          B     '                   #HIGH_PRECISION C   #NUM_REDSHIFTS D   #KMAX E   #K_PER_LOGINT F   #REDSHIFTS G                �                               C                                �                               D                               �                              E               
                �                               F                               �                              G     �                        
  p          p �           p �                                      �   �                           H     
       (             #   
  p          & p        p 
           p 
                                      �                               I     x      $                   �                               J     0       �      %       #REIONIZATIONHISTORY K                  @  @                          K     '0                    #TAU_START L   #TAU_COMPLETE M   #AKTHOM N   #FHE O   #WINDOWVARMID P   #WINDOWVARDELTA Q                �                              L                
                �                              M               
                �                              N               
                �                              O               
                �                              P                
                �                              Q     (          
                �                               R     �      &                   �                               S     �      '                   �                               T     �      (                   �                              U     �      )   
                �                              V     �      *   
                �                              W     �      +   
                �                              X     �      ,   
                �                              Y     �      -   
                �                              Z     �      .   
                                                [     	                 	                 
ף;            0.005#         @                                   \                    #CAMB_PK ]             D @                               ]     �              #MATTERPOWERDATA       �         fn#fn    �   ;   b   uapp(NONLINEAR    �   @   J  MODELPARAMS    9  @   J  TRANSFER )   y  �      MATTERPOWERDATA+TRANSFER /   %  H   a   MATTERPOWERDATA%NUM_K+TRANSFER /   m  H   a   MATTERPOWERDATA%NUM_Z+TRANSFER 0   �  �   a   MATTERPOWERDATA%LOG_KH+TRANSFER 3   I  �   a   MATTERPOWERDATA%REDSHIFTS+TRANSFER 2   �  �   a   MATTERPOWERDATA%MATPOWER+TRANSFER /   �  �   a   MATTERPOWERDATA%DDMAT+TRANSFER 6   5  �   a   MATTERPOWERDATA%NONLIN_RATIO+TRANSFER '   �  6     CAMBPARAMS+MODELPARAMS /   	  H   a   CAMBPARAMS%WANTCLS+MODELPARAMS 4   _	  H   a   CAMBPARAMS%WANTTRANSFER+MODELPARAMS 3   �	  H   a   CAMBPARAMS%WANTSCALARS+MODELPARAMS 3   �	  H   a   CAMBPARAMS%WANTTENSORS+MODELPARAMS 3   7
  H   a   CAMBPARAMS%WANTVECTORS+MODELPARAMS 1   
  H   a   CAMBPARAMS%DOLENSING+MODELPARAMS 1   �
  H   a   CAMBPARAMS%NONLINEAR+MODELPARAMS -     H   a   CAMBPARAMS%MAX_L+MODELPARAMS 4   W  H   a   CAMBPARAMS%MAX_L_TENSOR+MODELPARAMS 1   �  H   a   CAMBPARAMS%MAX_ETA_K+MODELPARAMS 8   �  H   a   CAMBPARAMS%MAX_ETA_K_TENSOR+MODELPARAMS .   /  H   a   CAMBPARAMS%OMEGAB+MODELPARAMS .   w  H   a   CAMBPARAMS%OMEGAC+MODELPARAMS .   �  H   a   CAMBPARAMS%OMEGAV+MODELPARAMS .     H   a   CAMBPARAMS%OMEGAN+MODELPARAMS *   O  H   a   CAMBPARAMS%H0+MODELPARAMS ,   �  H   a   CAMBPARAMS%TCMB+MODELPARAMS +   �  H   a   CAMBPARAMS%YHE+MODELPARAMS 7   '  H   a   CAMBPARAMS%NUM_NU_MASSLESS+MODELPARAMS 6   o  H   a   CAMBPARAMS%NUM_NU_MASSIVE+MODELPARAMS :   �  H   a   CAMBPARAMS%NU_MASS_SPLITTINGS+MODELPARAMS ;   �  H   a   CAMBPARAMS%NU_MASS_EIGENSTATES+MODELPARAMS <   G  �   a   CAMBPARAMS%NU_MASS_DEGENERACIES+MODELPARAMS 9   �  �   a   CAMBPARAMS%NU_MASS_FRACTIONS+MODELPARAMS @     H   a   CAMBPARAMS%SCALAR_INITIAL_CONDITION+MODELPARAMS ;   �  H   a   CAMBPARAMS%OUTPUTNORMALIZATION+MODELPARAMS <     H   a   CAMBPARAMS%ACCURATEPOLARIZATION+MODELPARAMS 2   W  H   a   CAMBPARAMS%ACCURATEBB+MODELPARAMS <   �  H   a   CAMBPARAMS%ACCURATEREIONIZATION+MODELPARAMS 7   �  H   a   CAMBPARAMS%MASSIVENUMETHOD+MODELPARAMS 1   /  h   a   CAMBPARAMS%INITPOWER+MODELPARAMS 0   �  �      INITIALPOWERPARAMS+INITIALPOWER 3   H  H   a   INITIALPOWERPARAMS%NN+INITIALPOWER 3   �  �   a   INITIALPOWERPARAMS%AN+INITIALPOWER 6   ,  �   a   INITIALPOWERPARAMS%N_RUN+INITIALPOWER 4   �  �   a   INITIALPOWERPARAMS%ANT+INITIALPOWER 4   d  �   a   INITIALPOWERPARAMS%RAT+INITIALPOWER ;      H   a   INITIALPOWERPARAMS%K_0_SCALAR+INITIALPOWER ;   H  H   a   INITIALPOWERPARAMS%K_0_TENSOR+INITIALPOWER ?   �  �   a   INITIALPOWERPARAMS%SCALARPOWERAMP+INITIALPOWER -   ,  h   a   CAMBPARAMS%REION+MODELPARAMS 0   �  �      REIONIZATIONPARAMS+REIONIZATION =   P  H   a   REIONIZATIONPARAMS%REIONIZATION+REIONIZATION B   �  H   a   REIONIZATIONPARAMS%USE_OPTICAL_DEPTH+REIONIZATION 9   �  H   a   REIONIZATIONPARAMS%REDSHIFT+REIONIZATION ?   (  H   a   REIONIZATIONPARAMS%DELTA_REDSHIFT+REIONIZATION 9   p  H   a   REIONIZATIONPARAMS%FRACTION+REIONIZATION >   �  H   a   REIONIZATIONPARAMS%OPTICAL_DEPTH+REIONIZATION .      i   a   CAMBPARAMS%RECOMB+MODELPARAMS 2   i  �      RECOMBINATIONPARAMS+RECOMBINATION @   �  H   a   RECOMBINATIONPARAMS%RECFAST_FUDGE+RECOMBINATION C   @  H   a   RECOMBINATIONPARAMS%RECFAST_FUDGE_HE+RECOMBINATION C   �  H   a   RECOMBINATIONPARAMS%RECFAST_HESWITCH+RECOMBINATION 0   �  d   a   CAMBPARAMS%TRANSFER+MODELPARAMS +   4  �      TRANSFERPARAMS+MODELPARAMS :   �  H   a   TRANSFERPARAMS%HIGH_PRECISION+MODELPARAMS 9     H   a   TRANSFERPARAMS%NUM_REDSHIFTS+MODELPARAMS 0   f  H   a   TRANSFERPARAMS%KMAX+MODELPARAMS 8   �  H   a   TRANSFERPARAMS%K_PER_LOGINT+MODELPARAMS 5   �  �   a   TRANSFERPARAMS%REDSHIFTS+MODELPARAMS >   �  �   a   CAMBPARAMS%INITIALCONDITIONVECTOR+MODELPARAMS 5   >  H   a   CAMBPARAMS%ONLYTRANSFERS+MODELPARAMS 1   �  i   a   CAMBPARAMS%REIONHIST+MODELPARAMS 1   �  �      REIONIZATIONHISTORY+REIONIZATION ;   �   H   a   REIONIZATIONHISTORY%TAU_START+REIONIZATION >   �   H   a   REIONIZATIONHISTORY%TAU_COMPLETE+REIONIZATION 8   +!  H   a   REIONIZATIONHISTORY%AKTHOM+REIONIZATION 5   s!  H   a   REIONIZATIONHISTORY%FHE+REIONIZATION >   �!  H   a   REIONIZATIONHISTORY%WINDOWVARMID+REIONIZATION @   "  H   a   REIONIZATIONHISTORY%WINDOWVARDELTA+REIONIZATION ,   K"  H   a   CAMBPARAMS%FLAT+MODELPARAMS .   �"  H   a   CAMBPARAMS%CLOSED+MODELPARAMS ,   �"  H   a   CAMBPARAMS%OPEN+MODELPARAMS .   ##  H   a   CAMBPARAMS%OMEGAK+MODELPARAMS ,   k#  H   a   CAMBPARAMS%CURV+MODELPARAMS )   �#  H   a   CAMBPARAMS%R+MODELPARAMS -   �#  H   a   CAMBPARAMS%KSIGN+MODELPARAMS ,   C$  H   a   CAMBPARAMS%TAU0+MODELPARAMS ,   �$  H   a   CAMBPARAMS%CHI0+MODELPARAMS !   �$  u       MIN_KH_NONLINEAR *   H%  U       NONLINEAR_GETNONLINRATIOS 2   �%  ]   a   NONLINEAR_GETNONLINRATIOS%CAMB_PK 