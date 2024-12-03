"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[6458],{4036:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>l,contentTitle:()=>r,default:()=>c,frontMatter:()=>a,metadata:()=>o,toc:()=>d});var s=n(5893),i=n(1151);const a={title:"Translate Headers and Labels",sidebar_position:1,description:"sample page",custom_edit_url:null},r=void 0,o={id:"Overview/Use Cases/Translate Headers and Labels",title:"Translate Headers and Labels",description:"sample page",source:"@site/docs/01-Overview/Use Cases/02-Translate Headers and Labels.mdx",sourceDirName:"01-Overview/Use Cases",slug:"/Overview/Use Cases/Translate Headers and Labels",permalink:"/solution-openpages-wxai-translation/Overview/Use Cases/Translate Headers and Labels",draft:!1,unlisted:!1,editUrl:null,tags:[],version:"current",sidebarPosition:1,frontMatter:{title:"Translate Headers and Labels",sidebar_position:1,description:"sample page",custom_edit_url:null},sidebar:"tutorialSidebar",previous:{title:"Translate User Fields",permalink:"/solution-openpages-wxai-translation/Overview/Use Cases/Translate User Fields"},next:{title:"Prepare",permalink:"/solution-openpages-wxai-translation/Prepare"}},l={},d=[{value:"Method 1: Translate Individual Items from the UI.",id:"method-1-translate-individual-items-from-the-ui",level:2},{value:"Where it&#39;s used",id:"where-its-used",level:3},{value:"How to Implement",id:"how-to-implement",level:3},{value:"Method 2: Translate Configuration Export",id:"method-2-translate-configuration-export",level:2},{value:"Where it&#39;s used",id:"where-its-used-1",level:3},{value:"How to Implement",id:"how-to-implement-1",level:3}];function h(e){const t={a:"a",admonition:"admonition",h2:"h2",h3:"h3",hr:"hr",img:"img",li:"li",ol:"ol",p:"p",strong:"strong",ul:"ul",...(0,i.a)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)("div",{style:{borderBottom:"1px solid #000",marginTop:"20px",marginBottom:"20px"}}),"\n",(0,s.jsx)(t.p,{children:'Aside from the default feature of using the "Locale" to translate the headers and labels, translatations can be carried out through two methods:'}),"\n",(0,s.jsxs)(t.ul,{children:["\n",(0,s.jsxs)(t.li,{children:[(0,s.jsx)(t.strong,{children:"Method 1:"})," Translate Individual Items from the UI."]}),"\n",(0,s.jsxs)(t.li,{children:[(0,s.jsx)(t.strong,{children:"Method 2:"})," Translating and reuploading the configuration exports within OpenPages."]}),"\n"]}),"\n",(0,s.jsx)(t.h2,{id:"method-1-translate-individual-items-from-the-ui",children:"Method 1: Translate Individual Items from the UI."}),"\n",(0,s.jsx)(t.hr,{}),"\n",(0,s.jsx)(t.h3,{id:"where-its-used",children:"Where it's used"}),"\n",(0,s.jsx)(t.p,{children:'Translations on the headers and labels can be carried out on the OpenPages platform through the "System Configuration" and/or "Application Text", etc. by using the connected transltion service.'}),"\n",(0,s.jsxs)(t.ol,{children:["\n",(0,s.jsxs)(t.li,{children:["\n",(0,s.jsx)(t.p,{children:'Navigate to the "Open Administrator Menu", select "System Configuration", and select either "Application Text" or "Object Text".'}),"\n",(0,s.jsx)(t.p,{children:(0,s.jsx)(t.img,{alt:"Object Text",src:n(9615).Z+"",width:"3206",height:"1838"})}),"\n"]}),"\n",(0,s.jsxs)(t.li,{children:["\n",(0,s.jsx)(t.p,{children:'Select one of the items in the list, and select "Auto Translate" in the pop-up window on the right to carry out the relevant translations. Click "Done".'}),"\n",(0,s.jsx)(t.p,{children:(0,s.jsx)(t.img,{alt:"Object Text 2",src:n(99).Z+"",width:"3204",height:"1836"})}),"\n"]}),"\n"]}),"\n",(0,s.jsx)(t.h3,{id:"how-to-implement",children:"How to Implement"}),"\n",(0,s.jsx)(t.p,{children:"The UI-based translation can be configured one of two ways:"}),"\n",(0,s.jsxs)(t.ol,{children:["\n",(0,s.jsxs)(t.li,{children:["\n",(0,s.jsx)(t.p,{children:"OpenPages watsonx.ai translation service"}),"\n",(0,s.jsxs)(t.p,{children:["To translate using one of the two native translation services (NeuralSeek/watsonx.ai), complete the setup process as described in the ",(0,s.jsx)(t.a,{href:"https://www.ibm.com/docs/en/openpages/9.0.0?topic=integrations-translation-services",children:"product documentation"}),". See ",(0,s.jsx)(t.a,{href:"../../Create/Translate%20User%20Fields/Native%20Translation%20Service",children:"here"})," for more information."]}),"\n"]}),"\n",(0,s.jsxs)(t.li,{children:["\n",(0,s.jsx)(t.p,{children:"Custom translation service"}),"\n",(0,s.jsxs)(t.p,{children:["To translate the necessary headers/labels with wx.ai an API wrapper is required to simulate the API response that is expected from the NeuralSeek configuration within OpenPages. Navigate ",(0,s.jsx)(t.a,{href:"../../Create/Translate%20User%20Fields/Custom%20API%20Endpoint",children:"here"})," for a walkthrough on how to implement the API wrapper."]}),"\n"]}),"\n"]}),"\n",(0,s.jsx)(t.h2,{id:"method-2-translate-configuration-export",children:"Method 2: Translate Configuration Export"}),"\n",(0,s.jsx)(t.hr,{}),"\n",(0,s.jsx)(t.h3,{id:"where-its-used-1",children:"Where it's used"}),"\n",(0,s.jsx)(t.p,{children:"Another way to translate the headers/labels for the OpenPages platform is by exporting the necessary configuration files, translating the data, and then re-importing the translated configuration file back into the OpenPages platform."}),"\n",(0,s.jsx)(t.p,{children:(0,s.jsx)(t.strong,{children:"Export the configuration"})}),"\n",(0,s.jsxs)(t.ol,{children:["\n",(0,s.jsxs)(t.li,{children:["\n",(0,s.jsx)(t.p,{children:'Navigate to the "Open Administrator Menu", select "System Migration", and choose "Export Configuration".'}),"\n",(0,s.jsx)(t.p,{children:(0,s.jsx)(t.img,{alt:"Export Config",src:n(4376).Z+"",width:"3456",height:"1812"})}),"\n"]}),"\n",(0,s.jsxs)(t.li,{children:["\n",(0,s.jsx)(t.p,{children:'Select "Add Items +", choose "Application Text" from the list and select all necessary items and select "Select"'}),"\n",(0,s.jsx)(t.p,{children:(0,s.jsx)(t.img,{alt:"Export Config 2",src:n(6959).Z+"",width:"3202",height:"1836"})}),"\n"]}),"\n"]}),"\n",(0,s.jsx)(t.p,{children:(0,s.jsx)(t.strong,{children:"Translate the exported configuration file with wx.ai"})}),"\n",(0,s.jsxs)(t.p,{children:["Use the parse_config.py file under the Translate Configuration Export folder of ",(0,s.jsx)(t.a,{href:"https://github.com/ibm-client-engineering/solution-openpages-wxai-translation",children:"this repository"})," to parse the XML config file and translate the non-translated fields."]}),"\n",(0,s.jsx)(t.admonition,{type:"info",children:(0,s.jsx)(t.p,{children:"The code as it is currently written is intended for translating English fields to Japanese. The program will check using a regex parser for Japanese text to verify whether or not that field needs to be translated. You may need to implement a different system to check if a field should be translated."})}),"\n",(0,s.jsx)(t.admonition,{type:"info",children:(0,s.jsx)(t.p,{children:"The configuration export file also includes code for downloading an excel file from OpenPages to use as a preset dictionary to check for BEFORE using generative AI to translate. You may or may not want to use a system like this."})}),"\n",(0,s.jsx)(t.p,{children:(0,s.jsx)(t.strong,{children:"Import the configuration"})}),"\n",(0,s.jsxs)(t.ol,{children:["\n",(0,s.jsxs)(t.li,{children:["\n",(0,s.jsx)(t.p,{children:'Navigate to the "Open Administrator Menu", select "System Migration", and choose "Import Configuration".'}),"\n",(0,s.jsx)(t.p,{children:(0,s.jsx)(t.img,{alt:"Import Config",src:n(427).Z+"",width:"3204",height:"1646"})}),"\n"]}),"\n"]}),"\n",(0,s.jsx)(t.h3,{id:"how-to-implement-1",children:"How to Implement"}),"\n",(0,s.jsxs)(t.ul,{children:["\n",(0,s.jsx)(t.li,{children:"Work in progress"}),"\n"]})]})}function c(e={}){const{wrapper:t}={...(0,i.a)(),...e.components};return t?(0,s.jsx)(t,{...e,children:(0,s.jsx)(h,{...e})}):h(e)}},4376:(e,t,n)=>{n.d(t,{Z:()=>s});const s=n.p+"assets/images/OP_ConfigExport-266dba46c5b5cf9c7078dc59b939bee9.png"},6959:(e,t,n)=>{n.d(t,{Z:()=>s});const s=n.p+"assets/images/export-bc9f8048f39a35320ca437b3cebf78c7.png"},427:(e,t,n)=>{n.d(t,{Z:()=>s});const s=n.p+"assets/images/import-f90c7dd6eb7f0ad8a9a2c55982145c91.png"},9615:(e,t,n)=>{n.d(t,{Z:()=>s});const s=n.p+"assets/images/nav_headers-25037db50abef0bad69ff2b7f9149c85.png"},99:(e,t,n)=>{n.d(t,{Z:()=>s});const s=n.p+"assets/images/translate_headers-7802e860d7c471769373488987b75dbe.png"},1151:(e,t,n)=>{n.d(t,{Z:()=>o,a:()=>r});var s=n(7294);const i={},a=s.createContext(i);function r(e){const t=s.useContext(a);return s.useMemo((function(){return"function"==typeof e?e(t):{...t,...e}}),[t,e])}function o(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:r(e.components),s.createElement(a.Provider,{value:t},e.children)}}}]);