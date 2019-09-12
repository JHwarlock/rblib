#-*-encoding:utf-8 -*-  
"""
from pychartdir import * 
setLicenseCode("SXZVFNRN9MZ9L8LGA0E2B1BB")#这个是注册，防止有logo产生
data = [21, 18, 15, 12, 8, 24,17,18,19,20,11,17,23,29,11,79,100,58] 
labels = ["Labor", "Licenses", "Taxes", "Legal", "Facilities", "Production","Labor1", "Licenses1", "Taxes1", "Legal1", "Facilities1", "Production1","Labor2", "Licenses2", "Taxes2", "Legal2", "Facilities2", "Production2",]  
colors = ['0x66aaee', '0xeebb22', '0xbbbbbb', '0x8844ff', '0xdd2222', '0x009900'] *3
c = PieChart(600, 320)
#c.setBackground(c.linearGradientColor(0, 0, 0, 100, '0x99ccff', '0xffffff'),'0x888888') 
#c.setRoundedFrame()
c.setDropShadow() 

#设置图表标题。标题上方留出16像素空白  
c.addTitle("Pie Chart With Legend Demonstration", "timesbi.ttf", 18).setMargin2(0, 0,  16, 0)
c.setPieSize(300, 300, 300)  
# 画3D饼图，厚度为25像素  
c.set3D(80) 
c.setData(data, labels)
#设置扇区颜色  
c.setColors2(DataColor, colors) 
#给扇区设置梯度阴影效果 
c.setSectorStyle(LocalGradientShading)  
# 使用侧边标签布局方法，使标签距离饼图16像素  
c.setLabelLayout(SideLayout,16)  
c.setLabelFormat("{={sector}+1}")  
#设置标签样式。  
c.setLabelStyle("arialbd.ttf", 10).setBackground(Transparent, '0x444444')
#生成图例栏对象b，左侧中部坐标(300,175)，并设置字体arial，加粗倾斜，字体大小。  
b = c.addLegend(330, 175, 1, "arialbi.ttf", 10)  
#设置左对齐。  
b.setAlignment(Left) 
#设置图例背景和圆角效果  
b.setBackground(Transparent, '0x444444')  
b.setRoundedCorners()  
# 设置图例栏边距16像素，图例内容行间距5像素  
b.setMargin(16)  
b.setKeySpacing(0, 5) 
#设置图例各项内容图标无边缘。  
b.setKeyBorder(SameAsMainColor) 
b.setText(\
	#表示图例实体的各部分要靠顶对齐  
	#显示扇区编号，然后往后tab一个22像素距离  
	"<*block,valign=top*>{={sector}+1}.<*advanceTo=22*>"\
	#120像素块来展示label  
	"<*block,width=120*>{label}<*/*>"\
	#40像素块来展示百分比，而且是右对齐  
	"<*block,width=40,halign=right*>{percent}<*/*>%")
c.makeChart("legendpie2.png")
"""

