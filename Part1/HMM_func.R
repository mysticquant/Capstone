library("depmixS4")
HMfit <- function(inp.ret){
  
  ret.xts <- inp.ret
  states <- 2
  
  model.spec <- depmix(eval(parse(text=names(ret.xts)))~1,nstates=states,data=ret.xts)
  model.fit <- fit(model.spec)
  model.posterior <- posterior(model.fit)
  sum.frame <- summary(model.fit)
  normal.idx <- which(sum.frame[,1]>0 & sum.frame[,1]==min(sum.frame[,1]))
  event.idx <- index(sum.frame)[-normal.idx]
  
  HM.fit <- list(
    normal.state = normal.idx,
    event.state = event.idx,
    turb =ret.xts,
    spec =model.spec,
    fit = model.fit,
    summary = summary(model.fit),
    normal.prob = model.posterior[,normal.idx+1],
    normal.class = model.posterior[,'state']==normal.idx,
    event.prob = model.posterior[,event.idx+1],
    event.class = model.posterior[,'state']==event.idx
  )
  
  HM.fit$df = data.frame(Time.Idx=index(HM.fit$turb),
                         Turbulence=HM.fit$turb,
                         Normal.Prob=HM.fit$normal.prob,
                         Event.Prob=HM.fit$event.prob,
                         Normal.Class=ifelse(HM.fit$normal.class==T,1,0),
                         Event.Class=ifelse(HM.fit$event.class==T,1,0),
                         row.names=NULL)
  return(HM.fit$df)
}