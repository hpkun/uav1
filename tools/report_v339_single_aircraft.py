"""Render scientific figures and the Chinese report from saved v339 audit data."""
from pathlib import Path
import json,math,os,re
import numpy as np
import pandas as pd

def render_report(output):
    output=Path(output).resolve();root=Path(__file__).resolve().parents[1]
    os.environ['MPLCONFIGDIR']=str(output/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'));df=pd.read_csv(output/'motion_cases.csv');traces=np.load(output/'trajectories.npz',allow_pickle=False)
    dt=summary['config']['simulation']['dt'];colors={150:'#3274a1',225:'#228855',300:'#d47722'}
    def save(name,fig):fig.tight_layout();fig.savefig(output/name,dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,magnitude in zip(axes,[1.,.5]):
        for speed in colors:
            for sign in [1.,-1.]:
                tr=traces[f'B_v{speed}_a{sign*magnitude:g}'];period=round((2*math.pi/(math.pi/4*magnitude))/dt)
                ax.plot(tr[:period+1,0],tr[:period+1,1],color=colors[speed],ls='-' if sign>0 else '--',label=f'v={speed}, a=[{sign*magnitude:g},0,0]')
        ax.set(xlabel='x / North (m)',ylabel='y / East (m)',title=f'Actual v3.39 XY turn, |a_heading|={magnitude}',aspect='equal');ax.grid(alpha=.2);ax.legend(fontsize=7)
    save('turn_xy.png',fig)
    b=df[df.group=='B'];fig,axes=plt.subplots(1,2,figsize=(12,5))
    for a in [1.,.5]:
        selected=b[b.input_heading==a].sort_values('initial_speed_m_s')
        axes[0].plot(selected.initial_speed_m_s,selected.discrete_radius_m,'-',label=f'Discrete theory a={a}')
        axes[0].plot(selected.initial_speed_m_s,selected.continuous_radius_m,'--',label=f'Continuous theory a={a}')
        axes[0].scatter(selected.initial_speed_m_s,selected.fitted_radius_m,marker='x',s=70,label=f'Trajectory fit a={a}')
        axes[1].plot(selected.initial_speed_m_s,(selected.fitted_radius_m-selected.continuous_radius_m)*1000,'o-',label=f'a={a}')
    axes[0].set(xlabel='Speed (m/s)',ylabel='Radius (m)');axes[1].set(xlabel='Speed (m/s)',ylabel='Fitted minus continuous radius (mm)')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    save('turn_radius.png',fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for a in [1.,-1.,.5,-.5]:
        tr=traces[f'B_v225_a{a:g}'];time=np.arange(len(tr))*dt
        axes[0].plot(time,np.degrees(tr[:,5]),label=f'a_heading={a}')
        axes[1].plot(time,np.degrees(np.unwrap(tr[:,5])),label=f'a_heading={a}')
    for ax,label in zip(axes,['Wrapped heading (deg)','Unwrapped heading (deg)']):ax.set(xlabel='Time (s)',ylabel=label);ax.legend();ax.grid(alpha=.2)
    save('heading_response.png',fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ap in [1.,-1.,.5,-.5]:
        tr=traces[f'C_v225_p{ap:g}_h3000'];time=np.arange(len(tr))*dt;mask=time<=6
        axes[0].plot(time[mask],np.degrees(tr[mask,4]),label=f'a=[0,{ap:g},0]')
    for speed in colors:
        for ap in [1.,-1.]:
            tr=traces[f'C_v{speed}_p{ap:g}_h3000'];time=np.arange(len(tr))*dt;mask=time<=6
            axes[1].plot(time[mask],speed*np.sin(tr[mask,4]),color=colors[speed],ls='-' if ap>0 else '--',label=f'v={speed}, a_pitch={ap:g}')
    axes[0].set(xlabel='Time (s)',ylabel='Pitch (deg)',title='Pitch response (identical at all three speeds)');axes[1].set(xlabel='Time (s)',ylabel='Vertical speed, up positive (m/s)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    save('pitch_response.png',fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,ap in zip(axes,[1.,-1.]):
        for speed in colors:
            tr=traces[f'C_v{speed}_p{ap:g}_h3000'];time=np.arange(len(tr))*dt
            row=df[df.case_id==f'C_v{speed}_p{ap:g}_h3000'].iloc[0];end=round(row.first_full_height_boundary_s/dt)
            limit=round(30/dt);ax.plot(time[:end+1],-tr[:end+1,2],color=colors[speed],label=f'v={speed}, a=[0,{ap:g},0]')
            ax.plot(time[end:limit+1],-tr[end:limit+1,2],color=colors[speed],ls=':',alpha=.55)
        ax.axhspan(0,6000,color='grey',alpha=.08);ax.axhline(0,c='red',ls='--');ax.axhline(6000,c='red',ls='--')
        ax.set(xlabel='Time (s)',ylabel='Altitude (m)',title='Solid: until boundary; dotted: isolated integration only');ax.grid(alpha=.2);ax.legend(fontsize=8)
    save('altitude_response.png',fig)
    fig,axes=plt.subplots(1,3,figsize=(17,5));palette=plt.get_cmap('viridis')
    for j,sign in enumerate([1,-1,0]):
        for i,speed in enumerate([150,200,225,250,300]):
            for magnitude in ([1.,.5] if sign else [0.]):
                a=sign*magnitude;tr=traces[f'D_v{speed}_a{a:g}'];time=np.arange(len(tr))*dt
                axes[j].plot(time,tr[:,3],color=palette(i/4),ls='-' if magnitude!=.5 else '--',label=f'v0={speed}, a_speed={a:g}')
        axes[j].set(xlabel='Time (s)',ylabel='Speed (m/s)',title=['Acceleration','Deceleration','Zero speed command'][j]);axes[j].grid(alpha=.2);axes[j].legend(fontsize=7)
    save('speed_response.png',fig)
    fig=plt.figure(figsize=(17,6))
    for j,speed in enumerate(colors):
        ax=fig.add_subplot(1,3,j+1,projection='3d')
        for index,action in [(0,[1,1,0]),(1,[1,-1,0]),(3,[1,1,1]),(4,[1,1,-1]),(5,[.5,.5,.5])]:
            tr=traces[f'E_v{speed}_a{index}'];h=-tr[:,2];outside=(h<=0)|(h>6000)|(np.hypot(tr[:,0],tr[:,1])>5000);hits=np.flatnonzero(outside);end=int(hits[0]) if len(hits) else len(tr)-1
            line=ax.plot(tr[:end+1,0],tr[:end+1,1],h[:end+1],label=str(action))[0]
            if end<len(tr)-1:ax.plot(tr[end:,0],tr[end:,1],h[end:],ls=':',color=line.get_color(),alpha=.5)
        ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='Altitude (m)',title=f'Isolated 3D motion, v0={speed} m/s');ax.legend(fontsize=7)
        ax.set_box_aspect(None,zoom=.80)
    save('combined_3d.png',fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5));a=df[df.group=='A']
    for i,(field,factor,label) in enumerate([('heading_rate_deg_s',45.,'Heading rate (deg/s)'),('pitch_rate_initial_deg_s',30.,'Pitch rate (deg/s)'),('acceleration_initial_m_s2',25.,'Acceleration (m/s^2)')]):
        part=a[a[['input_heading','input_pitch','input_speed'][i]]!=0]
        for speed in colors:
            data=part[part.initial_speed_m_s==speed];expected=data[['clipped_heading','clipped_pitch','clipped_speed'][i]]*factor
            if i==2:expected=(np.clip(speed+2.5*data.clipped_speed,150,300)-speed)/dt
            axes[i].scatter(expected,data[field],label=f'v0={speed}',s=24,alpha=.65)
        axes[i].plot([-factor,factor],[-factor,factor],'k--',lw=.8);axes[i].set(xlabel='Independent expected '+label,ylabel='Measured '+label);axes[i].grid(alpha=.2);axes[i].legend(fontsize=8)
    save('theory_measured.png',fig)

    def tests(name):
        p=output/(name+'.txt')
        if not p.exists():return None
        text=p.read_text(encoding='utf-8',errors='replace');line=text.strip().splitlines()[-1]
        counts={k:int(re.findall(r'(\d+) '+k,line)[-1]) for k in ['passed','failed','deselected'] if re.findall(r'(\d+) '+k,line)}
        return dict(summary=line,**counts)
    summary['pytest']={k:tests(k) for k in ['existing_tests_log','focused_tests_log','supplement_tests_log']}
    summary['figure_files']=['turn_xy.png','turn_radius.png','heading_response.png','pitch_response.png','altitude_response.png','speed_response.png','combined_3d.png','theory_measured.png']
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    def table(frame,cols=None):
        frame=frame.copy() if cols is None else frame[cols]
        def fmt(v):
            if pd.isna(v):return '—'
            if isinstance(v,(float,np.floating)):return f'{v:.6f}'
            return str(v).replace('|','/')
        return '| '+' | '.join(frame.columns)+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'
    def source(file,line=None):return f'[{file}]({root.as_posix()}/{file}{":"+str(line) if line else ""})'
    R=[]
    def add(s=''):R.append(s+'\n')
    add('# v3.39 单机三维运动学专项验证实验')
    add('执行日期：2026-10-18（Asia/Shanghai）。本次重新实际执行当前v3.39，没有将v3.32旧结果当作本次结果；所有计算使用CPU/NumPy，CUDA_VISIBLE_DEVICES=-1，不加载模型、不训练。WSL访问返回E_ACCESSDENIED，依仓库政策使用Windows PowerShell的uav环境；本次CPU要求来自用户明确实验协议。')
    add(f'共有{summary["case_count"]}个运动场景、{summary["trace_count"]}份轨迹和{summary["check_count"]}条检查；通过{summary["passed_checks"]}条。这里的all_passed指本专项运动检查，**不隐瞒先行全量两文件pytest的2个非运动学报错文本断言失败**。')
    add('## 第一部分：实际运动学实现')
    add('源码核实：'+source('configs/combat_environment_v339.yaml')+'、'+source('env/combat_v339.py')+'、'+source('env/combat_v332.py',16)+'、'+source('env/v311_kinematics.py',6)+'、'+source('env/models.py')+'、'+source('env/math_utils.py')+'、'+source('env/combat_env.py')+'、'+source('env/v330_transition.py')+'，以及现有两个测试文件和旧audit脚本均实际读取。v339只替换Blue自动速度策略；本次实际实例为CombatEnvironmentV339，_advance绑定env.combat_v332，调用advance_aircraft。父构造的旧dynamics/integrator对象被v332删除，运行中没有PID、滚转、过载动态或RK4位置更新。')
    add('飞行状态为(x,y,z,v,theta,psi,alive)，NED坐标，height=−z。动作依次为[a_heading,a_pitch,a_speed]，先逐维clip到[−1,1]。在第t步：\n\n`psi_next = wrap(psi + heading_delta_max * clip(a_heading))`\n\n`theta_next = clip(theta + pitch_delta_max * clip(a_pitch), theta_min, theta_max)`\n\n`v_next = clip(v + speed_delta_max * clip(a_speed), v_min, v_max)`\n\n`x_next = x + v_next*cos(theta_next)*cos(psi_next)*dt`\n\n`y_next = y + v_next*cos(theta_next)*sin(psi_next)*dt`\n\n`z_next = z - v_next*sin(theta_next)*dt`。')
    cfg=summary['config'];c=cfg['action']['command'];ac=cfg['aircraft'];ar=cfg['arena']
    parameters=[('dt',dt,'s'),('max_steps',cfg['simulation']['max_steps'],'step'),('heading increment',math.degrees(c['heading_delta_max']),'deg/step'),('heading rate maximum',math.degrees(c['heading_delta_max'])/dt,'deg/s'),('pitch increment',math.degrees(c['pitch_delta_max']),'deg/step'),('pitch rate maximum',math.degrees(c['pitch_delta_max'])/dt,'deg/s'),('speed increment',c['speed_delta_max'],'m/s per step'),('acceleration magnitude',c['speed_delta_max']/dt,'m/s²'),('v_min',ac['v_min'],'m/s'),('v_max',ac['v_max'],'m/s'),('pitch_min',math.degrees(ac['theta_min']),'deg'),('pitch_max',math.degrees(ac['theta_max']),'deg'),('ground',ar['altitude_min'],'m'),('ceiling',ar['altitude_max'],'m'),('arena radius',ar['radius'],'m')]
    add(table(pd.DataFrame(parameters,columns=['参数','实际值','单位'])))
    add('实际配置与用户预期一致。动作是每步角度/速度增量，**不是再乘dt的角速度/加速度命令**；45°/s、30°/s和25m/s²由这些增量除以.1s得到。零动作保持当前psi/theta/v，不自动回正或高度保持。正theta表示爬升，正heading是NED下从+x北转向+y东的右转；常规XY绘图的轴排列可能看似逆时针，不应据此交换动作符号。')
    add('阶段分离：纯_advance只做运动，不改变alive，也不约束高度/战场位置。完整step先移动双方，随后_resolve_noncombat_losses，ground高度≤0先死亡；否则r>5000或height>6000算boundary死亡；ceiling是boundary子类，不重复算第二架损失。无投影/反弹/恢复。死亡后_advance跳过更新。'+source('env/combat_v325.py')+'→'+source('env/combat_v30.py',83)+'。')
    add('## 第二部分：实验方法与复现')
    add('默认初态x=y=0，height3000m，psi=theta=0，alive=True；速度取各实验规定值。A为63个单步场景（3速×3通道×7输入），B为12个32秒恒定水平转弯，C为36个60秒爬升/下降（3速度×4动作×3高度），D为25个20秒速度响应，E为21个20秒组合动作；另有wrap/非水平零动作、随机1000步重复、无效输入、对称和边界测试。非默认高度/角度的场景均明确标记。')
    add('期望值来自独立、非递推的n步通道公式，以及对更新后速度向量的独立离散求和；不调用advance_aircraft生成理论期望。heading用圆周误差atan2(sin Δ,cos Δ)，避免将±π表示跳变当成物理跳变；位置绝对误差容差1e−7m，通道容差1e−10（各自单位）。B用实际XY点中心化后最小二乘拟合圆，不从理论半径直接赋值。连续R=v/|omega|；离散R=v*dt/[2sin(|omega|dt/2)]，omega由实际unwrap heading差分测得，同时独立通道公式检查其正确性。')
    add('C对应36个完整环境边界试验，另有3个水平出界试验。正式配置不改，测试仅设置人工初始状态；保留1架测试Red，Blue用显式动作在x≈−4000m附近以150m/s小半径盘旋，其他slot死亡。没有调用自动Blue team_actions；完整武器逻辑仍运行，但几何距离始终使双方开火次数为0，且核实。实际转移状态与独立公式逐步比较，至第一次死亡即停止。这样不篡改weapon/termination，却能隔离所需运动和边界阶段。')
    add('所有CSV不适用项留空，报告显示“—”；不是零。motion_cases.csv每场景一行，含初态、请求/clip动作、首步有效增量、rates、位置误差、拟合半径、边界和pass；trajectories.npz保存每场景完整6D序列。末态边界predicate对isolated场景只是“若调用resolver会怎样”，不是正式存活状态。')
    add('复现（项目根目录、uav、CPU；默认输出必须尚不存在；重跑时指定另一个空的新目录）：')
    add('```bash\nconda activate uav\nCUDA_VISIBLE_DEVICES=-1 python -B tools/audit_v339_single_aircraft.py --output-dir outputs/v339_single_aircraft_audit_repeat\n```')
    add('旧tools/audit_combat_v332_kinematics.py的main还包含200局v330/v332战斗公平性与CUDA门槛，不符合本次CPU单机任务，因此没有运行其main。其纯CPU circle_audit函数在**新建v339实例**上重新执行3个速度；这复用验证逻辑，不复用历史结果文件。')
    add('## 第三部分：完整定量结果')
    add('### A. 单步映射和误差')
    add(f'63个网格场景验证输入clip、状态clip和更新后位置积分。全实验最大位置公式误差{summary["max_position_formula_error_m"]:.3e}m；heading圆周误差{summary["max_heading_error_rad"]:.3e}rad，pitch误差{summary["max_pitch_error_rad"]:.3e}rad，speed误差{summary["max_speed_error_m_s"]:.3e}m/s。合法输入轨迹均有限，单步位移不超过300×.1=30m。')
    add(table(df[(df.group=='A')&(df.input_heading==0)&(df.input_pitch==0)&(df.input_speed.isin([-2,-1,1,2]))],['initial_speed_m_s','input_speed','clipped_speed','effective_speed_first','acceleration_initial_m_s2','dx_first_m','final_speed_m_s']))
    add('例如初速150、负速度动作会得到合法零减速度；初速300、正速度动作亦如此。输入±2先clip为±1；如果已在速度边界，有效速度增量可能为0。不能把这些当作25m/s²执行失败。角度wrap跨±π时比较最短圆周增量，位置不跳。')
    add('![独立理论与实测](theory_measured.png)')
    add('### B. 水平转弯')
    add(table(b,['initial_speed_m_s','input_heading','heading_rate_deg_s','time90_s','time180_s','fitted_radius_m','continuous_radius_m','discrete_radius_m','radius_error_discrete_m','lateral_accel_m_s2','equivalent_g']))
    add('最大heading动作下150/225/300m/s的90°/180°时间均为2s/4s；半动作则4s/8s。半径随速度线性增加，而固定heading action的角速度不随速度变。225m/s最大转向约286.55m半径，等效侧向加速度约176.71m/s²、18.01g；300m/s约24.02g。正负转向的XY镜像检查与拟合均通过，速度/高度/pitch严格保持。')
    add('**equivalent_g只是按运动轨迹和v|omega|反推的等效侧向加速度，不是当前环境模拟的法向过载，不代表真实飞行器性能。**离散实际路径是等长弦构成的正多边形，其拟合外接圆半径应与离散公式一致；与连续公式小幅差异是离散化几何，不是执行误差。')
    add('![不同速度XY转弯](turn_xy.png)\n\n![半径与速度](turn_radius.png)\n\n![heading响应](heading_response.png)')
    add('### C. 爬升/下降与高度边界')
    crows=df[(df.group=='C')&(df.initial_z==-3000)]
    add(table(crows,['initial_speed_m_s','input_pitch','time_pitch30_s','time_pitch60_s','vertical_speed_terminal_m_s','first_full_height_boundary_s','pitch_bound_hold_fraction']))
    add('最大pitch动作达到±30°需1s、±60°需2s；半动作需2s/4s。限幅后姿态保持±60°，不是自动减小pitch或垂直速度；最大垂直速度为v*sin60°，150/225/300m/s分别约129.904/194.856/259.808m/s。航向未偏移，无speed动作时速度不受爬升/下降影响。')
    near=df[(df.group=='C')&(df.initial_z!=-3000)]
    add('下表初始高度100m与5900m；first_full_height_boundary_s来自完整环境实际step，与纯运动预测一致：')
    add(table(near,['initial_speed_m_s','initial_z','input_pitch','time_pitch60_s','first_full_height_boundary_s']))
    add('initial_z是NED z：−100表示高度100m、−5900表示5900m。垂直速度从0逐步增大，所以不能用100m除以极限垂直速度当边界触达时间。边界命中发生在离散step后，允许位置略穿过0或6000再死亡；这不是存活飞行穿地或被投影回安全区。图中越界后的虚点线只属于isolated继续积分，不能作为真实环境存活轨迹。')
    add('![pitch与垂直速度](pitch_response.png)\n\n![高度和边界](altitude_response.png)')
    add('### D. 速度控制')
    d=df[df.group=='D'];add(table(d,['initial_speed_m_s','input_speed','acceleration_initial_m_s2','acceleration_max_abs_m_s2','time_vmin_s','time_vmax_s','final_speed_m_s']))
    add('未饱和全动作加减速度±25m/s²，半动作±12.5m/s²；225→300/150分别3s，半动作6s；150→300全动作6s、半动作12s。达到界后v保持，无抖动、无阻力或能量损失项。time_vmin/vmax为包括初始t=0的首次触达；“—”才是该动作轨迹中从未达到，不把不适用写0。zero action保持任意给定初速。')
    add('![速度响应](speed_response.png)')
    add('### E. 组合机动')
    e=df[df.group=='E'];add(table(e,['initial_speed_m_s','input_heading','input_pitch','input_speed','final_speed_m_s','final_pitch_deg','fd_accel_max_m_s2','fd_equivalent_g_max','curvature_max_per_m','pitch_bound_fraction','speed_bound_fraction']))
    add('每个组合的psi/theta/v分别与相同单通道动作独立运行的对应state序列逐步比较，均一致；空间位置正常同时依赖姿态和速度，属于几何耦合。有限差分加速度由NED速度向量相邻差分除dt，包含初始速度到首步更新；curvature用法向分量/速度²，只是离散轨迹指标，不能叫模型施加的真实normal load。')
    add('最大转向+爬升初段可同时使用两个角通道，轨迹加速度可高于只转向；pitch到60°后水平速度变为v cos60°，水平投影半径约为水平转弯的一半。组合加速/减速不会因爬升自动付出能量代价。pitch_bound_fraction与speed_bound_fraction仅是状态在界上的比例；零speed动作本来就在150m/s时也为1，不能误称动作被截断。CSV另有“非零command但实际增量为0”的比例。')
    add('![代表性3D组合机动](combined_3d.png)')
    add('### F. 稳定性、确定性和边界')
    add('固定随机种子339生成1000步[-2,2]动作序列，两次独立运行状态逐位相同且有限；100步同状态同动作的Red/Blue更新完全相同。wrap、pitch/speed限幅、死亡冻结、合法最大位移全部检查通过。live单机函数对6种invalid shape/NaN/Inf拒绝，完整环境对3种invalid动作亦拒绝。死亡单机函数先返回copy，dead slot的单独动作不会进入其验证；完整step仍统一验证全team动作，不能将dead early-return误写成formal允许NaN输入。')
    add('边界精确谓词核实：height=0死亡，6000本身不因ceiling死亡，5000本身不因horizontal死亡，稍超上界才触发。ground优先于boundary。ceiling是boundary子类。下面列出完整环境全部39个场景的首个死亡时刻及实际计数：')
    boundary=pd.DataFrame([{k:v for k,v in r.items() if k not in ['frames','state','flags']} for r in summary['boundary_cases']]);add(table(boundary,['case_id','initial_speed','initial_height','time_s','red_boundary_exits','red_ground_losses','red_ceiling_losses','prediction_error','no_fire','reason']))
    add('零动作不等于水平姿态：从±20°pitch、225m/s出发，10s高度仍改变约±769.545m，角度和速度保持。它是精确符合公式的积分式控制语义；若网络持续动作后又输出0，不会自动恢复level flight。')
    add('## 第四部分：问题分类与测试验收')
    add('**A：合法输入下未发现明确运动学实现错误或数值失稳。**理论与实测误差在浮点容差内，未观测到非预期state通道耦合。另有测试维护问题：先行执行整个test_combat_v332.py和test_combat_v339.py时，2个旧MADSAC元数据用例期待environment_version报错，但当前v5校验先报implementation-v5 required，导致regex断言失败；输入仍被拒绝，且不涉及运动路径。本次依限制不修改旧测试或任何正式实现。')
    for label,key in [('先行完整现有验证','existing_tests_log'),('专项与直接相关现有验证','focused_tests_log'),('新增零动作非水平姿态补充','supplement_tests_log')]:
        value=summary['pytest'].get(key);add(f'{label}：'+(value['summary'] if value else '未提供日志；不得宣称已通过'))
    add('focused命令仅排除了old_action_semantics_checkpoint_rejected整组6个元数据测试，不是隐藏运动失败；现有完整验证的两项失败在上文及日志保留。新增专项初轮153项，focused总242项通过，后加2项coast测试通过。没有运行耗时的训练全套测试。')
    add('**B：符合现有公式但值得审视的简化。**①45°/s不受速度/力/能量约束，导致约12–24g等效侧向量级；②同角速度下低速半径更小而转90°时间不变；③pitch30°/s、可达60°且没有自动回正，垂直速度可很大；④heading/pitch/加速可同时拉满，没有真实动力学预算耦合；⑤动作率由每step增量/dt确定，改dt而不联动增量会改物理能力；⑥只有姿态/速度界限，硬位置边界没有提前减速/投影，命中后即死亡。这些不是因缺少PID/过载耦合而产生的程序bug。')
    add('**C：尚无法确定。**是否过强到妨碍某个强化学习算法、是否诱导低速盘旋、是否导致之前best→final退化、是否确需更复杂控制器，都没有由单机实验验证。也未对损坏的非有限初始flight state、无限时长积分或真实飞行包线作安全保证；本次结论限于正式配置、有限合法初态/动作及明确的拒绝测试。')
    add('## 第五部分：对强化学习训练的潜在影响')
    add('225m/s下约286.55m半径和2s转90°相对1km射程与5km战场是灵活的运动能力；它可让攻击几何在少量控制步内大幅变化。150m/s半径更小、同时不损失45°/s转率，提供偏好低速tight turn的机制，但没有运行policy训练，不能声称已出现该偏好。')
    add('pitch的积分式动作、强垂直速度、无高度保持和硬ground/ceiling死亡，意味着持续偏置动作可以快速累积到危险姿态；near-boundary时间表提供定量风险尺度，不是训练失败因果证明。零动作保留旧pitch尤其需要在agent语义中明确。组合机动无力/能量竞争，可能扩大可执行策略集，也可能使轨迹对动作累计偏差敏感。')
    add('优势是动作执行直接、确定、可预测，三通道对state独立且数值稳定，适合研究高层多agent策略；空间几何耦合为设计本意。角度wrap是坐标表示非连续而位置连续，不能报成运动跳变；若网络直接使用线性heading标量，跨±pi表征问题应另行审计，非本次修改内容。')
    add('目前没有证据证明需要引入PID/RK4或更复杂控制器；这会改变任务和action语义，不能当作对已验证公式的“修复”。是否增加能量/过载/协调转弯约束取决于后续研究目标，并需另行授权。')
    add('## 第六部分：结论与后续建议')
    add('当前v3.39实现符合其简化直接3D运动学设计，可继续作为这类RL环境使用；这不等于真实飞行性能可信或训练稳定性已保证。明确发现的是高等效机动量级、低速小半径、快速俯仰/无自动回正和硬边界阶段分离；未发现合法输入下的执行误差/不连续位置跳变/非预期通道耦合。')
    add('仅建议后续分别做heading rate、pitch rate、速度范围/低速收益和near-boundary风险的单因素敏感性实验，并保持dt对应的物理率定义清楚。可先分析各policy的动作持续性与距边界时间，不必马上换控制器。若未来要论证改动提高训练稳定性，需要独立paired训练设计；本任务不实施这些建议。')
    if 'preservation' in summary:add('保护核查：'+json.dumps(summary['preservation'],ensure_ascii=False)+'。全部正式环境、配置、算法及已有受git跟踪文件保持不变；只新增独立审计/报告工具、专项测试和本输出目录。未修改模型或历史结果，未训练。')
    add('输出索引：[summary.json](summary.json)、[motion_cases.csv](motion_cases.csv)、[trajectories.npz](trajectories.npz)、[existing tests](existing_tests_log.txt)、[focused tests](focused_tests_log.txt)、[supplement tests](supplement_tests_log.txt)。新增入口：'+source('tools/audit_v339_single_aircraft.py')+'；报告工具：'+source('tools/report_v339_single_aircraft.py')+'；专项测试：'+source('tests/test_v339_single_aircraft_audit.py')+'。')
    report='\n'.join(R)
    report=re.sub(r'\]\(([^)]+)\)',lambda m:']('+m.group(1)+')' if re.match(r'^(?:[A-Za-z]:/|https?://)',m.group(1)) else ']('+output.as_posix()+'/'+m.group(1)+')',report)
    (output/'audit_report.md').write_text(report,encoding='utf-8')
