"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[3722],{4246:(e,A,n)=>{n.r(A),n.d(A,{assets:()=>l,contentTitle:()=>i,default:()=>c,frontMatter:()=>a,metadata:()=>r,toc:()=>o});var t=n(5893),s=n(1151);const a={title:"Translate User Fields",sidebar_position:1,description:"sample page",custom_edit_url:null},i=void 0,r={id:"Overview/Use Cases/Translate User Fields",title:"Translate User Fields",description:"sample page",source:"@site/docs/01-Overview/Use Cases/01-Translate User Fields.mdx",sourceDirName:"01-Overview/Use Cases",slug:"/Overview/Use Cases/Translate User Fields",permalink:"/solution-openpages-wxai-translation/Overview/Use Cases/Translate User Fields",draft:!1,unlisted:!1,editUrl:null,tags:[],version:"current",sidebarPosition:1,frontMatter:{title:"Translate User Fields",sidebar_position:1,description:"sample page",custom_edit_url:null},sidebar:"tutorialSidebar",previous:{title:"Business Overview",permalink:"/solution-openpages-wxai-translation/Overview/Business Overview"},next:{title:"Translate Headers and Labels",permalink:"/solution-openpages-wxai-translation/Overview/Use Cases/Translate Headers and Labels"}},l={},o=[{value:"Where it&#39;s used",id:"where-its-used",level:3},{value:"How to Implement",id:"how-to-implement",level:3}];function d(e){const A={a:"a",em:"em",h3:"h3",img:"img",li:"li",ol:"ol",p:"p",...(0,s.a)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)("div",{style:{borderBottom:"1px solid #000",marginTop:"20px",marginBottom:"20px"}}),"\n",(0,t.jsx)(A.p,{children:"Once a translation service has been set up, user-inputted fields can be translated for transitory use. Individual fields must be selected one at a time for translation, and the output from the service will not be saved."}),"\n",(0,t.jsx)(A.h3,{id:"where-its-used",children:"Where it's used"}),"\n",(0,t.jsxs)(A.ol,{children:["\n",(0,t.jsxs)(A.li,{children:["\n",(0,t.jsx)(A.p,{children:"Navigate to any page within OpenPages where the translation button appears."}),"\n",(0,t.jsx)(A.p,{children:(0,t.jsx)(A.img,{alt:"translation button",src:n(7755).Z+"",width:"280",height:"68"})}),"\n"]}),"\n",(0,t.jsxs)(A.li,{children:["\n",(0,t.jsx)(A.p,{children:"Depening on what locale the user is in, the button will translate the fields to that target locale language."}),"\n",(0,t.jsxs)(A.p,{children:[(0,t.jsx)(A.em,{children:"Example below translating Janapese in an English locale"}),"\n",(0,t.jsx)(A.img,{alt:"translation button 2",src:n(1741).Z+"",width:"3200",height:"1830"})]}),"\n"]}),"\n"]}),"\n",(0,t.jsx)(A.h3,{id:"how-to-implement",children:"How to Implement"}),"\n",(0,t.jsx)(A.p,{children:"The UI-based translation can be configured one of two ways:"}),"\n",(0,t.jsxs)(A.ol,{children:["\n",(0,t.jsxs)(A.li,{children:["\n",(0,t.jsx)(A.p,{children:"OpenPages watsonx.ai translation service"}),"\n",(0,t.jsxs)(A.p,{children:["To translate using one of the two native translation services (NeuralSeek/watsonx.ai), complete the setup process as described in the ",(0,t.jsx)(A.a,{href:"https://www.ibm.com/docs/en/openpages/9.0.0?topic=integrations-translation-services",children:"product documentation"}),". See ",(0,t.jsx)(A.a,{href:"../../Create/Translate%20User%20Fields/Native%20Translation%20Service",children:"here"})," for more information."]}),"\n"]}),"\n",(0,t.jsxs)(A.li,{children:["\n",(0,t.jsx)(A.p,{children:"Custom translation service"}),"\n",(0,t.jsxs)(A.p,{children:["To translate the necessary headers/labels with wx.ai an API wrapper is required to simulate the API response that is expected from the NeuralSeek configuration within OpenPages. Navigate ",(0,t.jsx)(A.a,{href:"../../Create/Translate%20User%20Fields/Custom%20API%20Endpoint",children:"here"})," for a walkthrough on how to implement the API wrapper."]}),"\n"]}),"\n"]})]})}function c(e={}){const{wrapper:A}={...(0,s.a)(),...e.components};return A?(0,t.jsx)(A,{...e,children:(0,t.jsx)(d,{...e})}):d(e)}},1741:(e,A,n)=>{n.d(A,{Z:()=>t});const t=n.p+"assets/images/after_translate-3a1c10013efbb77243473d4c2d465496.png"},7755:(e,A,n)=>{n.d(A,{Z:()=>t});const t="data:image/png;base64,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"},1151:(e,A,n)=>{n.d(A,{Z:()=>r,a:()=>i});var t=n(7294);const s={},a=t.createContext(s);function i(e){const A=t.useContext(a);return t.useMemo((function(){return"function"==typeof e?e(A):{...A,...e}}),[A,e])}function r(e){let A;return A=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:i(e.components),t.createElement(a.Provider,{value:A},e.children)}}}]);