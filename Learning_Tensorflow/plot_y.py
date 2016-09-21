import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

y_model_np = np.array(pd.read_csv("y_model.csv",sep = ","))
y_pred_np = np.array(pd.read_csv("y_pred.csv",sep = ","))

f, ((ax1, ax2, ax3), (ax4, ax5, ax6),(ax7,ax8,ax9)) = plt.subplots(3, 3, sharex='col', sharey='row')
ax1.plot(y_model_np[0,:],label = "Model")
ax1.plot(y_pred_np[0,:], label = "Pred")
#ax1.set_title('Sharing x per column, y per row')
ax2.plot(y_model_np[1,:],label = "Model")
ax2.plot(y_pred_np[1,:], label = "Pred")

ax3.plot(y_model_np[2,:],label = "Model")
ax3.plot(y_pred_np[2,:], label = "Pred")

ax4.plot(y_model_np[3,:],label = "Model")
ax4.plot(y_pred_np[3,:], label = "Pred")

ax5.plot(y_model_np[4,:],label = "Model")
ax5.plot(y_pred_np[4,:], label = "Pred")
#ax1.set_title('Sharing x per column, y per row')
ax6.plot(y_model_np[5,:],label = "Model")
ax6.plot(y_pred_np[5,:], label = "Pred")

ax7.plot(y_model_np[6,:],label = "Model")
ax7.plot(y_pred_np[6,:], label = "Pred")

ax8.plot(y_model_np[7,:],label = "Model")
ax8.plot(y_pred_np[7,:], label = "Pred")

ax9.plot(y_model_np[8,:],label = "Model")
ax9.plot(y_pred_np[8,:], label = "Pred")

plt.legend()
plt.show()