import xml.etree.ElementTree as ET

def parse_nessus(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    hosts = []
    
    for report_host in root.iter('ReportHost'):
        
        host = {
            "ip": None,
            "hostname": None,
            "vulns": []
        }
        
        for tag in report_host.find('HostProperties'):
            if tag.get("name") == "host-ip":
                host["ip"] = tag.text
            if tag.get("name") == "hostname":
                host["hostname"] = tag.text

        for item in report_host.iter('ReportItem'):
            vuln = {
                "cve": item.findtext("cve"),
                "cvss": float(item.findtext("cvss_base_score") or 0),
                "port": item.get("port"),
                "service": item.get("svc_name")
            }
            host["vulns"].append(vuln)
        
        hosts.append(host)
    
    return hosts


if __name__ == "__main__":
    hosts = parse_nessus("data/sample.nessus")
    for host in hosts:
        print(host)