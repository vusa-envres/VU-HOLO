import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes=23):
        super(ConvNet, self).__init__()
        
        self.layer11 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.01))  
        
        self.layer21 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.01))      
        
        self.layer31 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),             
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.01))     
        self.layer41 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),    
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),    
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.01))  
        self.layer51 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),    
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),    
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.01))        

        self.layer12 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.01))  
        self.layer22 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.01))      
        self.layer32 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(), 
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),             
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.01))     
        self.layer42 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),    
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),    
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.01))  
        self.layer52 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),    
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),    
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.01))           
                
        self.fc0 = nn.Linear(4608, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
      
        
    def forward(self, X1, X2):
        x1 = self.layer11(X1)
        x1 = self.layer21(x1)
        x1 = self.layer31(x1)
        x1 = self.layer41(x1)
        x1 = self.layer51(x1)        
        x2 = self.layer12(X2)
        x2 = self.layer22(x2)
        x2 = self.layer32(x2)  
        x2 = self.layer42(x2) 
        x2 = self.layer52(x2)        
        x1 = x1.reshape(x1.size(0), -1)
        x2 = x1.reshape(x2.size(0), -1)
        x = torch.cat( (x1,x2), 1)
        x = self.fc0(x)
        x = F.relu(x)
        x = self.fc1(x)   
        x = F.relu(x)
        x = self.fc2(x)          
        return x      
